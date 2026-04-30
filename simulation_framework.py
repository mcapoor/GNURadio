import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as signal
from scipy.fft import fft
from collections import deque


# ============================================================================
# AUDIO AND SYNCHRONIZATION
# ============================================================================

def receive_audio(device, sample_rate):
    """
    Record audio from microphone until user stops.

    Args:
        device: Audio device index (None = default)
    Returns:

        audio_data: Recorded audio as numpy array
    """
    print("Recording... Press Ctrl+C to stop.")
    audio_buffer = deque(maxlen=sample_rate * 10)  # Keep last 10 seconds of audio

    try:
        while True:
            chunk = sd.rec(1024, samplerate=sample_rate, channels=1, device=device)
            sd.wait()
            audio_buffer.extend(chunk[:, 0])  # Add new samples to buffer
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    return np.array(audio_buffer)


def transmit_audio(waveform, device, sample_rate):
    """
    Transmit audio waveform through speakers.

    Args:
        waveform: Numpy array of audio samples to transmit
        device: Audio device index (None = default)
    """
    print("Transmitting audio...")
    sd.play(waveform, samplerate=sample_rate, device=device)
    sd.wait()



# ============================================================================
# CHANNEL SIMULATION
# ============================================================================

def identity_channel_model(signal, snr_db, sample_rate): return signal

def channel_model(tx_signal, sample_rate=44100, snr_db=20, delay_samples=0, freq_offset=0,
                  phase_jitter=0, doppler_factor=1.0):
    """
    Simulate a noisy channel with realistic impairments.

    Args:
        tx_signal: Transmitted signal
        snr_db: Signal-to-noise ratio in dB
        sample_rate: Sample rate in Hz
        delay_samples: Number of samples to delay (propagation delay)
        freq_offset: Frequency offset in Hz (oscillator mismatch)
        phase_jitter: Phase jitter standard deviation in radians
        doppler_factor: Doppler shift factor (1.0 = no shift)

    Returns:
        rx_signal: Received signal after passing through channel
    """
    signal_power = np.mean(np.abs(tx_signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Add propagation delay
    if delay_samples > 0:
        rx_signal = np.concatenate([np.zeros(delay_samples), tx_signal])
    else:
        rx_signal = tx_signal.copy()

    # Add white gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(rx_signal))
    rx_signal = rx_signal + noise

    # Add frequency offset
    # TODO: this is not correct
    if freq_offset != 0:
        t = np.arange(len(rx_signal)) / sample_rate
        phase_drift = 2 * np.pi * freq_offset * t
        rx_signal = rx_signal * np.cos(phase_drift)

    # Add Doppler shift
    if doppler_factor != 1.0:
        indices = np.arange(len(rx_signal)) / doppler_factor
        indices = np.clip(indices, 0, len(rx_signal) - 1)
        rx_signal = np.interp(indices, np.arange(len(rx_signal)), rx_signal)

    # Add phase jitter (small random phase shifts)
    if phase_jitter > 0:
        phase_noise = np.random.normal(0, phase_jitter, len(rx_signal))
        rx_signal = rx_signal * np.exp(1j * phase_noise)

    return rx_signal

def identity_channel_model(signal, snr_db, sample_rate): return signal


# ============================================================================
# CONTINUOUS BIT STREAM GENERATOR
# ============================================================================

def bit_stream_generator(bernoulli_p=0.5, chunk_size=100):
    """
    Infinite bit stream generator that yields random bits in chunks.

    Args:
        bernoulli_p: Probability of 1 in Bernoulli distribution
        chunk_size: Number of bits to yield per iteration

    Yields:
        Bit strings of length chunk_size
    """
    from scipy.stats import bernoulli
    while True:
        bits = bernoulli.rvs(bernoulli_p, size=chunk_size)
        bit_string = ''.join(map(str, bits))
        yield bit_string

class StreamingTransceiver:
    """Maintains state for streaming transmission/reception with real-time updates."""

    def __init__(self, transmitter, receiver, channel_model_fn, sample_rate, bits_per_symbol = 2, snr_db=20, audio_device=None):
        self.transmitter = transmitter
        self.receiver = receiver
        self.channel_model_fn = channel_model_fn
        self.sample_rate = sample_rate
        self.bits_per_symbol = bits_per_symbol
        self.snr_db = snr_db
        self.audio_device = audio_device

        # Streaming accumulators
        self.tx_bits_buffer = ""
        self.rx_bits_buffer = ""
        self.tx_symbols_buffer = np.array([], dtype=complex)
        self.rx_symbols_buffer = np.array([], dtype=complex)
        self.tx_signal_buffer = np.array([], dtype=float)
        self.rx_signal_buffer = np.array([], dtype=float)

        # Statistics
        self.total_bits_processed = 0
        self.total_errors = 0
        self.running_ber_history = []
        self.snr_history = []
        self.phase_history = []  # Track PLL phase over time

    def process_chunk(self, tx_bits, max_buffer_size=10000):
        """
        Process a chunk of bits through the channel.

        Args:
            tx_bits: Bit string to transmit
            max_buffer_size: Max symbols to keep in buffer (for memory efficiency)
        """
        # Skip if no bits to process
        if not tx_bits:
            return

        # Transmit
        tx_signal = self.transmitter.transmit_bits(tx_bits)
        tx_symbols = self.transmitter.bits_to_symbols(tx_bits)

        # Channel
        rx_signal = self.channel_model_fn(tx_signal)

        # Receive
        result = self.receiver.receive_bits(rx_signal)
        if result is None:
            rx_bits = ""
            rx_symbols = np.array([], dtype=complex)
        else:
            rx_bits, rx_symbols = result

        # Track PLL phase estimate
        phase_estimate = self.receiver.get_phase_estimate()
        self.phase_history.append(np.degrees(phase_estimate))  # Convert to degrees for visualization

        # Accumulate data (keep buffer size bounded)
        self.tx_bits_buffer += tx_bits
        self.rx_bits_buffer += rx_bits
        self.tx_symbols_buffer = np.append(self.tx_symbols_buffer, tx_symbols)
        self.rx_symbols_buffer = np.append(self.rx_symbols_buffer, rx_symbols)
        self.tx_signal_buffer = np.append(self.tx_signal_buffer, tx_signal)
        self.rx_signal_buffer = np.append(self.rx_signal_buffer, rx_signal)

        # Trim buffers to max size (keep newest data)
        if len(self.tx_symbols_buffer) > max_buffer_size:
            trim_idx = len(self.tx_symbols_buffer) - max_buffer_size

            self.tx_bits_buffer = self.tx_bits_buffer[trim_idx * self.bits_per_symbol:]
            self.rx_bits_buffer = self.rx_bits_buffer[trim_idx * self.bits_per_symbol:]
            self.tx_symbols_buffer = self.tx_symbols_buffer[trim_idx:]
            self.rx_symbols_buffer = self.rx_symbols_buffer[trim_idx:]
            self.tx_signal_buffer = self.tx_signal_buffer[trim_idx * self.transmitter.samples_per_symbol:]
            self.rx_signal_buffer = self.rx_signal_buffer[trim_idx * self.receiver.samples_per_symbol:]

        # Update statistics
        bits_to_compare = min(len(self.tx_bits_buffer), len(self.rx_bits_buffer))
        if bits_to_compare > 0:
            tx_arr = np.array(list(self.tx_bits_buffer[:bits_to_compare]), dtype=int)
            rx_arr = np.array(list(self.rx_bits_buffer[:bits_to_compare]), dtype=int)
            errors = np.sum(tx_arr != rx_arr)
            ber = errors / bits_to_compare
            self.running_ber_history.append(ber)

        return {
            'tx_bits': tx_bits,
            'rx_bits': rx_bits,
            'tx_symbols': tx_symbols,
            'rx_symbols': rx_symbols,
            'tx_signal': tx_signal,
            'rx_signal': rx_signal
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_animated_transceiver_plot(streamer, bit_stream_gen, update_interval_ms=1000, max_frames=None):
    """
    Create animated plot showing real-time transmission/reception.

    Args:
        streamer: StreamingTransceiver instance
        bit_stream_gen: Generator yielding bit chunks
        update_interval_ms: Milliseconds between animation frames
        max_frames: Maximum frames to animate (None = infinite)
    """
    from matplotlib.animation import FuncAnimation

    # Get modulation name from transmitter
    def get_modulation_name():
        """Determine modulation type name."""
        try:
            # Try to get bits_per_symbol from modulation object
            if hasattr(streamer.transmitter, 'modulation') and hasattr(streamer.transmitter.modulation, 'bits_per_symbol'):
                bps = streamer.transmitter.modulation.bits_per_symbol
            else:
                bps = streamer.bits_per_symbol

            # Map bits per symbol to modulation name
            mapping = {1: 'BPSK', 2: 'QPSK', 3: '8-PSK', 4: '16-QAM'}
            return mapping.get(bps, f'{2**bps}-QAM')
        except:
            return 'Unknown'

    modulation_name = get_modulation_name()

    # Create figure and axes (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Real-Time Transceiver ({modulation_name} @ adaptive SNR)', fontsize=14, fontweight='bold')

    # Adjust spacing to prevent title overlap
    plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)

    frame_count = [0]

    # Animation state
    anim_state = {'paused': False, 'zoom_duration': 1.0}  # zoom_duration in seconds

    def on_key_press(event):
        """Handle keyboard zoom controls."""
        if event.key == '+' or event.key == '=':
            # Zoom in (decrease time window)
            anim_state['zoom_duration'] = max(0.001, anim_state['zoom_duration'] * 0.8)
        elif event.key == '-' or event.key == '_':
            # Zoom out (increase time window)
            anim_state['zoom_duration'] = min(5.0, anim_state['zoom_duration'] * 1.25)
        elif event.key == '0':
            # Reset to default
            anim_state['zoom_duration'] = 1.0

    def update_frame(frame):
        """Update plots with new chunk of data."""
        try:
            # Skip processing new data if paused, but still update plots
            if not anim_state['paused']:
                # Get next bit chunk
                bit_chunk = next(bit_stream_gen)

                # Process through channel
                streamer.process_chunk(bit_chunk)

            # Clear axes
            for ax in axes.flat:
                ax.clear()

            # Plot 1: Time-domain signals (zoom support)
            display_duration = anim_state['zoom_duration']  # Adjustable zoom
            display_samples = int(display_duration * streamer.sample_rate)

            if len(streamer.tx_signal_buffer) > 0 and len(streamer.rx_signal_buffer) > 0:
                # Get last 1 second of data (or all if less than 1 second available)
                tx_display = streamer.tx_signal_buffer[-display_samples:] if len(streamer.tx_signal_buffer) >= display_samples else streamer.tx_signal_buffer
                rx_display = streamer.rx_signal_buffer[-display_samples:] if len(streamer.rx_signal_buffer) >= display_samples else streamer.rx_signal_buffer

                # Ensure both signals have the same length for plotting
                min_len = min(len(tx_display), len(rx_display))
                tx_display = tx_display[:min_len]
                rx_display = rx_display[:min_len]

                # Create time vector for displayed data
                t_display = np.arange(len(tx_display)) / streamer.sample_rate

                axes[0, 0].plot(t_display, tx_display, linewidth=0.5, label='TX', alpha=0.8)
                axes[0, 0].plot(t_display, rx_display, linewidth=0.5, alpha=0.6, label='RX')
                axes[0, 0].set_title(f'Time Domain Signals ({display_duration:.3f}s zoom - +/- to adjust, 0 to reset)')
                axes[0, 0].set_xlabel('Time (s)')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Running BER
            if streamer.running_ber_history:
                axes[0, 1].semilogy(range(len(streamer.running_ber_history)),
                                   streamer.running_ber_history, 'r-o', markersize=4)
                axes[0, 1].set_title(f'Running BER (SNR={streamer.snr_db}dB)')
                axes[0, 1].set_xlabel('Chunk #')
                axes[0, 1].set_ylabel('Bit Error Rate')
                axes[0, 1].grid(True, alpha=0.3, which='both')
                latest_ber = streamer.running_ber_history[-1]
                axes[0, 1].text(0.95, 0.05, f'Latest BER: {latest_ber:.4f}',
                               transform=axes[0, 1].transAxes, ha='right', va='bottom',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Plot 2.5: Bit comparison (TX vs RX)
            max_bits_display = 20  # Show most recent 20 bits
            if len(streamer.tx_bits_buffer) > 0 and len(streamer.rx_bits_buffer) > 0:
                bits_to_show = min(max_bits_display, len(streamer.tx_bits_buffer), len(streamer.rx_bits_buffer))
                tx_bits_short = np.array([int(b) for b in streamer.tx_bits_buffer[-bits_to_show:]])
                rx_bits_short = np.array([int(b) for b in streamer.rx_bits_buffer[-bits_to_show:]])

                # Plot TX and RX bits
                axes[0, 2].plot(range(bits_to_show), tx_bits_short, 'b-o', markersize=3, alpha=0.6, label='TX')
                axes[0, 2].plot(range(bits_to_show), rx_bits_short, 'r-s', markersize=3, alpha=0.6, label='RX')

                # Highlight errors
                errors = (tx_bits_short != rx_bits_short)
                error_indices = np.where(errors)[0]
                if len(error_indices) > 0:
                    axes[0, 2].scatter(error_indices, tx_bits_short[error_indices],
                                      color='red', s=100, marker='x', linewidths=2, label='Errors', zorder=5)

                axes[0, 2].set_title(f'Bit Comparison (most recent {bits_to_show} bits)')
                axes[0, 2].set_xlabel('Bit Index')
                axes[0, 2].set_ylabel('Bit Value')
                axes[0, 2].set_ylim(-0.2, 1.2)
                axes[0, 2].set_yticks([0, 1])
                axes[0, 2].grid(True, alpha=0.3, axis='y')
                axes[0, 2].legend(loc='upper right', fontsize=8)

                # Add error count
                error_count = np.sum(errors)
                axes[0, 2].text(0.02, 0.98, f'Errors: {error_count}/{bits_to_show}',
                               transform=axes[0, 2].transAxes, ha='left', va='top',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=9)

            # Plot 3: TX Constellation
            if len(streamer.tx_symbols_buffer) > 0:
                axes[1, 0].scatter(streamer.tx_symbols_buffer.real,
                                  streamer.tx_symbols_buffer.imag,
                                  alpha=0.5, s=15, c='blue')
                axes[1, 0].set_title(f'TX Constellation ({len(streamer.tx_symbols_buffer)} symbols)')
                axes[1, 0].set_xlabel('I')
                axes[1, 0].set_ylabel('Q')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].axis('equal')


            # Plot 4: RX Constellation
            if len(streamer.rx_symbols_buffer) > 0:
                axes[1, 1].scatter(streamer.rx_symbols_buffer.real,
                                  streamer.rx_symbols_buffer.imag,
                                  alpha=0.5, s=15, c='red')
                axes[1, 1].set_title(f'RX Constellation ({len(streamer.rx_symbols_buffer)} symbols)')
                axes[1, 1].set_xlabel('I')
                axes[1, 1].set_ylabel('Q')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].axis('equal')


            # Plot 5: Phase Lock Loop estimate
            if len(streamer.phase_history) > 0:
                axes[1, 2].plot(range(len(streamer.phase_history)), streamer.phase_history, 'g-o', markersize=3)
                axes[1, 2].set_title('PLL Phase Estimate')
                axes[1, 2].set_xlabel('Chunk #')
                axes[1, 2].set_ylabel('Phase (degrees)')
                axes[1, 2].grid(True, alpha=0.3)
                latest_phase = streamer.phase_history[-1]
                axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target (0°)')
                axes[1, 2].legend(fontsize=8)
                axes[1, 2].text(0.98, 0.02, f'Current: {latest_phase:.1f}°',
                               transform=axes[1, 2].transAxes, ha='right', va='bottom',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

            # Plot 6: TX Frequency Domain (FFT)
            if len(streamer.tx_signal_buffer) > 0:
                tx_fft = np.abs(fft(streamer.tx_signal_buffer))
                tx_freqs = np.fft.fftfreq(len(streamer.tx_signal_buffer), 1/streamer.sample_rate)

                # Only positive frequencies
                positive_idx = tx_freqs >= 0
                tx_freqs = tx_freqs[positive_idx]
                tx_fft = tx_fft[positive_idx]

                # Normalize
                tx_fft = tx_fft / np.max(tx_fft) if np.max(tx_fft) > 0 else tx_fft

                axes[2, 0].semilogy(tx_freqs, tx_fft + 1e-10, linewidth=0.8, color='blue')
                axes[2, 0].set_title(f'TX Frequency Response ({len(streamer.tx_signal_buffer)} samples)')
                axes[2, 0].set_xlabel('Frequency (Hz)')
                axes[2, 0].set_ylabel('Magnitude (dB)')
                axes[2, 0].grid(True, alpha=0.3, which='both')
                axes[2, 0].set_xlim(0, streamer.sample_rate / 2)

            # Plot 7: RX Frequency Domain (FFT)
            if len(streamer.rx_signal_buffer) > 0:
                rx_fft = np.abs(fft(streamer.rx_signal_buffer))
                rx_freqs = np.fft.fftfreq(len(streamer.rx_signal_buffer), 1/streamer.sample_rate)

                # Only positive frequencies
                positive_idx = rx_freqs >= 0
                rx_freqs = rx_freqs[positive_idx]
                rx_fft = rx_fft[positive_idx]

                # Normalize
                rx_fft = rx_fft / np.max(rx_fft) if np.max(rx_fft) > 0 else rx_fft

                axes[2, 1].semilogy(rx_freqs, rx_fft + 1e-10, linewidth=0.8, color='orange')
                axes[2, 1].set_title(f'RX Frequency Response ({len(streamer.rx_signal_buffer)} samples)')
                axes[2, 1].set_xlabel('Frequency (Hz)')
                axes[2, 1].set_ylabel('Magnitude (dB)')
                axes[2, 1].grid(True, alpha=0.3, which='both')
                axes[2, 1].set_xlim(0, streamer.sample_rate / 2)

            frame_count[0] += 1
            fig.suptitle(f'Real-Time Transceiver ({modulation_name} @ {streamer.snr_db}dB) - Frame {frame_count[0]}',
                        fontsize=14, fontweight='bold')
            plt.show()

        except Exception as e:
            print(f"\n❌ ERROR in animation frame {frame_count[0]}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Create animation (max_frames=None for infinite streaming)
    anim = FuncAnimation(fig, update_frame, interval=update_interval_ms,
                        repeat=False, frames=max_frames, cache_frame_data=False)

    # Connect keyboard handler for zoom controls
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show(block=False)

    return fig, anim

def calculate_ber(tx_bits, rx_bits):
    """
    Calculate bit error rate.

    Args:
        tx_bits: Transmitted bit string
        rx_bits: Received bit string

    Returns:
        (ber, bit_errors, total_bits)
    """
    tx_array = np.array(list(tx_bits), dtype=int)
    rx_array = np.array(list(rx_bits), dtype=int)

    # Truncate to same length
    min_len = min(len(tx_array), len(rx_array))
    tx_array = tx_array[:min_len]
    rx_array = rx_array[:min_len]

    bit_errors = np.sum(tx_array != rx_array)
    ber = bit_errors / min_len if min_len > 0 else 1.0

    return ber, bit_errors, min_len


def plot_closed_loop_signals(tx_signal, rx_signal, tx_symbols, rx_symbols, carrier_freq, sample_rate):
    """
    Real-time signal display of transmitted and received signals with constellations.

    Args:
        tx_signal: Transmitted passband signal
        rx_signal: Received passband signal
        tx_symbols: Transmitted symbols
        rx_symbols: Received symbols
        carrier_freq: Carrier frequency in Hz
        sample_rate: Sample rate in Hz
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Time vectors
    t_tx = np.arange(len(tx_signal)) / sample_rate
    t_rx = np.arange(len(rx_signal)) / sample_rate

    # TX Signal
    display_duration = min(0.1, len(tx_signal) / sample_rate)
    display_samples = int(display_duration * sample_rate)
    axes[0, 0].plot(t_tx[:display_samples], tx_signal[:display_samples], linewidth=0.5, color='blue')
    axes[0, 0].set_title(f"Transmitted Signal (first {display_duration:.3f}s)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # RX Signal
    axes[0, 1].plot(t_rx[:display_samples], rx_signal[:display_samples], linewidth=0.5, color='orange')
    axes[0, 1].set_title(f"Received Signal (first {display_duration:.3f}s)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # TX Constellation
    axes[1, 0].scatter(tx_symbols.real, tx_symbols.imag, alpha=0.6, s=20, c='blue')
    axes[1, 0].set_title(f"TX Constellation ({len(tx_symbols)} symbols)")
    axes[1, 0].set_xlabel("I (In-phase)")
    axes[1, 0].set_ylabel("Q (Quadrature)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    axes[1, 0].set_xlim(-2, 2)
    axes[1, 0].set_ylim(-2, 2)

    # RX Constellation
    axes[1, 1].scatter(rx_symbols.real, rx_symbols.imag, alpha=0.6, s=20, c='red')
    axes[1, 1].set_title(f"RX Constellation ({len(rx_symbols)} symbols)")
    axes[1, 1].set_xlabel("I (In-phase)")
    axes[1, 1].set_ylabel("Q (Quadrature)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    axes[1, 1].set_xlim(-2, 2)
    axes[1, 1].set_ylim(-2, 2)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def plot_frequency_response(tx_signal, rx_signal, sample_rate, freq_range=(20, 20000)):
    """
    Real-time channel frequency response characterization.
    Displays power spectrum from freq_range[0] Hz to freq_range[1] Hz.

    Args:
        tx_signal: Transmitted signal
        rx_signal: Received signal
        sample_rate: Sample rate in Hz
        freq_range: Tuple of (min_freq, max_freq) in Hz
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute FFTs
    fft_tx = np.abs(fft(tx_signal))
    fft_rx = np.abs(fft(rx_signal))

    freqs_tx = np.fft.fftfreq(len(tx_signal), 1/sample_rate)
    freqs_rx = np.fft.fftfreq(len(rx_signal), 1/sample_rate)

    # Only positive frequencies
    positive_idx_tx = freqs_tx >= 0
    positive_idx_rx = freqs_rx >= 0
    freqs_tx = freqs_tx[positive_idx_tx]
    freqs_rx = freqs_rx[positive_idx_rx]
    fft_tx = fft_tx[positive_idx_tx]
    fft_rx = fft_rx[positive_idx_rx]

    # Filter to requested frequency range
    mask_tx = (freqs_tx >= freq_range[0]) & (freqs_tx <= freq_range[1])
    mask_rx = (freqs_rx >= freq_range[0]) & (freqs_rx <= freq_range[1])

    # TX Spectrum
    axes[0].semilogy(freqs_tx[mask_tx], fft_tx[mask_tx], linewidth=1)
    axes[0].set_title(f"TX Frequency Response ({freq_range[0]}-{freq_range[1]} Hz)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True, alpha=0.3, which='both')

    # RX Spectrum
    axes[1].semilogy(freqs_rx[mask_rx], fft_rx[mask_rx], linewidth=1, color='orange')
    axes[1].set_title(f"RX Frequency Response ({freq_range[0]}-{freq_range[1]} Hz)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def ber_vs_snr_sweep(transmitter, receiver, channel_model_fn, bit_rates=[50, 500, 5000],
                     snr_range=np.arange(0, 21, 3), num_bits=1000, sample_rate=44100, bits_per_symbol=2):
    """
    Graph of BER vs SNR for multiple bit rates.

    Args:
        transmitter: Transmitter instance
        receiver: Receiver instance
        channel_model_fn: Channel model function
        bit_rates: List of bit rates to test (bits per second)
        snr_range: Array of SNR values in dB
        num_bits: Number of bits to transmit for each test
        sample_rate: Sample rate in Hz
        bits_per_symbol: Number of bits per symbol (e.g. 2 for QPSK)
    """
    from scipy.stats import bernoulli

    results = {}

    for bit_rate in bit_rates:
        print(f"\n{'='*60}")
        print(f"Testing BER vs SNR at {bit_rate} bits/second")
        print(f"{'='*60}")

        bers = []

        for snr_db in snr_range:
            # Generate random bits
            tx_bits = ''.join(map(str, bernoulli.rvs(0.5, size=num_bits)))

            # Create transmitter for this bit rate
            tx = transmitter.__class__(
                modulation=transmitter.modulation,
                carrier_freq=transmitter.carrier_freq,
                sample_rate=sample_rate,
                bit_rate=bit_rate
            )

            # Transmit
            tx_signal = tx.transmit_bits(tx_bits)

            # Pass through channel with current SNR
            rx_signal = channel_model_fn(tx_signal, sample_rate=sample_rate, snr_db=snr_db)

            # Create receiver for this bit rate
            rx = receiver.__class__(
                modulation=receiver.modulation,
                carrier_freq=receiver.carrier_freq,
                sample_rate=sample_rate,
                bit_rate=bit_rate
            )

            # Receive
            result = rx.receive_bits(rx_signal)
            if result is None:
                rx_bits = ""
            else:
                rx_bits, rx_symbols = result

            # Calculate BER
            ber, _, _ = calculate_ber(tx_bits, rx_bits)
            bers.append(ber)

            print(f"SNR={snr_db:3d}dB: BER={ber:.4f}")

        results[bit_rate] = bers

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    for bit_rate, bers in results.items():
        ax.semilogy(snr_range, bers, 'o-', linewidth=2, markersize=6, label=f'{bit_rate} bps')

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER vs SNR for Different Bit Rates', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    return results


