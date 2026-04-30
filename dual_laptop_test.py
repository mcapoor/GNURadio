"""
Dual-Laptop OTA Transceiver Test 
=================================================

Run this script on two laptops to conduct over-the-air testing:

LAPTOP A (Transmitter):
  $ python dual_laptop_test.py --mode tx

LAPTOP B (Receiver):
  $ python dual_laptop_test.py --mode rx

Each laptop displays its own local graphs independently.

Or test on a single laptop with loopback:

"""

from modulation_framework import Transmitter, Receiver, nQAMModulation

from main import ( 
    Transmitter, UnifiedReceiver,
    nQAMModulation, 
    BFSKModulation, 
    CDMAModulation, CDMAReceiver
) 

import modulation_framework as milan

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import time
import sys
import threading
from collections import deque

plt.ioff()


# ============================================================================
# CONFIGURATION
# ============================================================================

CARRIER_FREQ = 4400
SAMPLE_RATE = 44_100
BITS_PER_SECOND = 5000

CHUNK_SIZE = 5000
PLOT_UPDATE_INTERVAL_MS = 100

RNG_SEED = 12345

# modulation = nQAMModulation(16)  
modulation = BFSKModulation(f0_offset=-10000, f1_offset=10000, fs=SAMPLE_RATE, spb=int(SAMPLE_RATE/BITS_PER_SECOND)) 

# ============================================================================
# TRANSMITTER NODE
# ============================================================================

def run_transmitter_node(audio_device=None, block=True):
    """Run transmitter that streams audio via speaker and displays local graphs."""
    print("\n" + "="*60)
    print("TRANSMITTER NODE")
    print("="*60)
    print(f"Transmitting via speaker at {CARRIER_FREQ} Hz carrier")
    print(f"Bit rate: {BITS_PER_SECOND} bps")
    print("="*60 + "\n")

    transmitter = Transmitter(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        bit_rate=BITS_PER_SECOND 
    )
    
    source = np.random.default_rng(RNG_SEED)
    data_lock = threading.Lock()
    stop_event = threading.Event()
    queue_lock = threading.Lock()
    audio_queue = deque()

    bit_list = []
    symbol_list = []
    signal_list = []

    def trim_buffers():
        if len(signal_list) > SAMPLE_RATE * 2:
            signal_list[:] = signal_list[-SAMPLE_RATE * 2:]

        while len(''.join(bit_list)) > 5000 and bit_list:
            bit_list.pop(0)

        if len(symbol_list) > 500:
            symbol_list[:] = symbol_list[-500:]

    def tx_worker():
        while stop_event.is_set() == False:
            bits = ''.join(str(b) for b in source.integers(0, 2, size=CHUNK_SIZE))
            tx_signal = transmitter.transmit_bits(bits)
            symbols = transmitter.bits_to_symbols(bits)

            with queue_lock:
                audio_queue.append(np.asarray(tx_signal, dtype=np.float32))
            # Open either a real OutputStream or the virtual device's output
            if isinstance(audio_device, VirtualLoopbackDevice):
                with audio_device.open_output(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=transmitter.samples_per_symbol) as out_stream:
                    buffer = audio_queue.popleft() if audio_queue else None

                    if buffer is not None:
                        out_stream.write(buffer)
            else:
                with sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    device=audio_device,
                    dtype='float32',
                    blocksize=transmitter.samples_per_symbol,
                ) as out_stream:
                    buffer = audio_queue.popleft() if audio_queue else None

                    if buffer is not None:
                        out_stream.write(buffer)
                
            with data_lock:
                bit_list.append(bits)
                symbol_list.extend(np.asarray(symbols).ravel().tolist())
                signal_list.extend(np.asarray(tx_signal).ravel().tolist())
                trim_buffers()

    anim_state = {'paused': False, 'zoom_duration': 1.0}
    
    def update_plots(frame):
        """Generate one chunk and update plots."""
        nonlocal bit_list, symbol_list, signal_list
        
        # Initialize snapshot if it doesn't exist
        if not hasattr(update_plots, 'snapshot'):
            update_plots.snapshot = {
                'signal': np.asarray(signal_list, dtype=float),
                'symbols': np.asarray(symbol_list, dtype=complex),
                'bits': list(bit_list)
            }
        
        # Handle pause state and update snapshot if not paused
        if not anim_state['paused']:
            with data_lock:
                signal_snapshot = np.asarray(signal_list, dtype=float)
                symbol_snapshot = np.asarray(symbol_list, dtype=complex)
                bit_snapshot = list(bit_list)
            update_plots.snapshot = {
                'signal': signal_snapshot,
                'symbols': symbol_snapshot,
                'bits': bit_snapshot
            }
        
        # Use snapshot (paused or not)
        signal_snapshot = update_plots.snapshot['signal']
        symbol_snapshot = update_plots.snapshot['symbols']
        bit_snapshot = update_plots.snapshot['bits']

        # Plot 1: Time-domain signals 
        display_duration = anim_state['zoom_duration']  # Adjustable zoom
        display_samples = min(int(display_duration * SAMPLE_RATE), len(signal_snapshot))
        
        ax = axes[0, 0]
        ax.clear()
        if display_samples > 0:
            t = np.arange(display_samples) / SAMPLE_RATE
            ax.plot(t, signal_snapshot[-display_samples:], linewidth=0.5, color='blue')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain (TX Signal)")
        ax.grid(True, alpha=0.3)

        # Plot 2: Frequency domain (FFT with windowing)
        ax = axes[0, 1]
        ax.clear()
        if len(signal_snapshot) >= SAMPLE_RATE:
            windowed_signal = signal_snapshot[-SAMPLE_RATE:] * np.hanning(SAMPLE_RATE)
            fft = np.abs(np.fft.fft(windowed_signal))
            freqs = np.fft.fftfreq(len(fft), 1 / SAMPLE_RATE)
            ax.plot(freqs[:len(freqs)//2], fft[:len(fft)//2], linewidth=0.5, color='blue')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.set_title("Frequency Domain (FFT)")
            ax.grid(True, alpha=0.3)

        # Plot 3: Last 20 bits
        ax = axes[1, 0]
        ax.clear()
        all_bits = ''.join(bit_snapshot)
        last_bits = all_bits[-20:] if len(all_bits) > 0 else ""
        if last_bits:
            bits_array = [int(b) for b in last_bits]
            colors = ['red' if b == 0 else 'blue' for b in bits_array]
            ax.bar(range(len(bits_array)), bits_array, color=colors)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Bit Index")
            ax.set_ylabel("Bit Value")
            ax.set_title(f"Source Bits (Last 20): {last_bits}")
            ax.set_xticks(range(len(bits_array)))

        # Plot 4: Constellation
        ax = axes[1, 1]
        ax.clear()
        if len(symbol_snapshot) > 0:
            ax.scatter(np.real(symbol_snapshot), np.imag(symbol_snapshot), alpha=0.6, s=50, color='blue')
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlabel("I (In-phase)")
            ax.set_ylabel("Q (Quadrature)")
            ax.set_title(f"Constellation ({len(symbol_snapshot)} symbols)")
            ax.grid(True, alpha=0.3)
            max_lim = 1.5
            ax.set_xlim(-max_lim, max_lim)
            ax.set_ylim(-max_lim, max_lim)

        # Add status overlay (pause/zoom info)
        status_text = f"Zoom: {anim_state['zoom_duration']:.3f}s"
        if anim_state['paused']:
            status_text = "⏸ PAUSED | " + status_text
        fig.text(0.02, 0.98, status_text, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        fig.text(0.02, 0.93, "[SPACE] Pause | [+/-] Zoom | [0] Reset", transform=fig.transFigure, fontsize=9,
                verticalalignment='top', color='gray')

        return axes.flat

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Transmitter Live View", fontsize=16, fontweight='bold')

    def on_close(_event):
        stop_event.set()
        
    def on_key_press(event):
        """Handle keyboard pause/zoom controls."""
        if event.key == ' ':
            # Toggle pause
            anim_state['paused'] = not anim_state['paused']
        elif event.key == '+' or event.key == '=':
            # Zoom in (decrease time window)
            anim_state['zoom_duration'] = max(0.001, anim_state['zoom_duration'] * 0.8)
        elif event.key == '-' or event.key == '_':
            # Zoom out (increase time window)
            anim_state['zoom_duration'] = min(5.0, anim_state['zoom_duration'] * 1.25)
        elif event.key == '0':
            # Reset to default
            anim_state['zoom_duration'] = 1.0
                    

    fig.canvas.mpl_connect("close_event", on_close)

    producer = threading.Thread(target=tx_worker, daemon=True)
    producer.start()
    
    # Give worker a moment to generate initial data
    time.sleep(0.2)

    anim = FuncAnimation(fig, update_plots, interval=PLOT_UPDATE_INTERVAL_MS, blit=False)
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    fig.canvas.mpl_connect('key_press_event', on_key_press)


    if block:
        try:
            plt.show(block=True)
        finally:
            stop_event.set()
            producer.join(timeout=1.0)
    else:
        return {
            'stop_event': stop_event,
            'thread': producer,
            'fig': fig,
            'anim': anim,
        }


   

# ============================================================================
# RECEIVER NODE
# ============================================================================

def run_receiver_node(audio_device=None, block=True):
    """Run receiver that records from mic and displays local graphs."""
    print("\n" + "="*60)
    print("RECEIVER NODE")
    print("="*60)
    print(f"Listening on mic at {CARRIER_FREQ} Hz carrier")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"RNG Seed (for known TX pattern): {RNG_SEED}")
    print("="*60 + "\n")

    receiver = Receiver(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        bit_rate=BITS_PER_SECOND
    )

    rx_bit_list = []
    tx_bit_list = []
    symbol_list = []
    signal_list = []
    chunk_count = 0
    data_lock = threading.Lock()
    stop_event = threading.Event()
    
    simulated_source = np.random.default_rng(RNG_SEED)
    capture_queue = deque()
    playback_queue = deque()
    playback_lock = threading.Lock()

    def trim_buffers():
        # Arbitrarily keep last 2 seconds audio, last 200 bits, and last 500 symbols 
        if len(signal_list) > SAMPLE_RATE * 2:
            signal_list[:] = signal_list[-SAMPLE_RATE * 2:]

        while len(''.join(rx_bit_list)) > 200:
            rx_bit_list.pop(0)

        while len(''.join(tx_bit_list)) > 200:
            tx_bit_list.pop(0)

        if len(symbol_list) > 500:
            symbol_list[:] = symbol_list[-500:]

    def input_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        capture_queue.append(np.asarray(indata[:, 0], dtype=np.float32).copy())

    def rx_worker():

        # If a virtual device was provided, use its input context manager which
        # will call `input_callback` with buffers from the virtual queue. Otherwise
        # use a real InputStream.
        if isinstance(audio_device, VirtualLoopbackDevice):
            with audio_device.open_input(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=input_callback, blocksize=4096):
                signal = np.zeros(0, dtype=np.float32)
                while not stop_event.is_set():
                    if capture_queue:
                        signal = np.concatenate((signal, capture_queue.popleft()))

                    # Process all available 1-second chunks
                    while len(signal) >= SAMPLE_RATE:
                        chunk = signal[:SAMPLE_RATE]
                        signal = signal[SAMPLE_RATE:]

                        result = receiver.receive_bits(chunk)
                        if result is None:
                            # store raw audio for plotting and continue
                            signal_list.extend(np.asarray(chunk).ravel().tolist())
                            trim_buffers()
                            continue

                        rx_bits, rx_symbols = result

                        tx_bits = ''.join(str(b) for b in simulated_source.integers(0, 2, size=len(rx_bits), dtype=int))

                        tx_bit_list.append(tx_bits)
                        rx_bit_list.append(rx_bits)

                        symbol_list.extend(np.asarray(rx_symbols).ravel().tolist())
                        signal_list.extend(np.asarray(chunk).ravel().tolist())
                        trim_buffers()
        else:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                device=audio_device,
                dtype='float32',
                callback=input_callback,
                blocksize=4096, #receiver.samples_per_symbol,
            ):
                signal = np.zeros(0, dtype=np.float32)
                while not stop_event.is_set():
                    if capture_queue:
                        signal = np.concatenate((signal, capture_queue.popleft()))
                        
                    # Process in chunks of 1 second or more
                    if len(signal) >= SAMPLE_RATE:
                        signal = signal[SAMPLE_RATE:]
     
                    rx_bits, rx_symbols = receiver.receive_bits(signal)
                    
                    tx_bits = ''.join(str(b) for b in simulated_source.integers(0, 2, size=len(rx_bits), dtype=int))
                    
                    tx_bit_list.append(tx_bits)
                    rx_bit_list.append(rx_bits)
                    
                    symbol_list.extend(np.asarray(rx_symbols).ravel().tolist())
                    signal_list.extend(np.asarray(signal).ravel().tolist())
                    trim_buffers()

    
    total_bits_compared = 0
    total_bit_errors = 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Receiver Live View", fontsize=16, fontweight='bold')

    anim_state = {'paused': False, 'zoom_duration': 1.0}
    static_snapshots = {'signal': np.array([]), 'symbols': np.array([]), 'rx_bits': [], 'tx_bits': []}

    def update_plots(frame):
        """Record one chunk and update plots."""
        nonlocal rx_bit_list, symbol_list, signal_list, chunk_count, tx_bit_list, total_bits_compared, total_bit_errors

        if anim_state['paused']:
            # Use static snapshots when paused
            rx_bits_snapshot = static_snapshots['rx_bits']
            tx_bits_snapshot = static_snapshots['tx_bits']
            symbol_snapshot = static_snapshots['symbols']
            signal_snapshot = static_snapshots['signal']
        else:
            with data_lock:
                rx_bits_snapshot = list(rx_bit_list)
                tx_bits_snapshot = list(tx_bit_list)
                symbol_snapshot = np.asarray(symbol_list, dtype=complex)
                signal_snapshot = np.asarray(signal_list, dtype=float)
            # Update static snapshots
            static_snapshots['rx_bits'] = rx_bits_snapshot
            static_snapshots['tx_bits'] = tx_bits_snapshot
            static_snapshots['symbols'] = symbol_snapshot
            static_snapshots['signal'] = signal_snapshot

        zoom_samples = int(SAMPLE_RATE * anim_state['zoom_duration'])

        chunk_count += 1

        # Plot 1: Time-domain signal
        ax = axes[0, 0]
        ax.clear()
        display_samples = min(zoom_samples, len(signal_snapshot))
        if display_samples > 0:
            t = np.arange(display_samples) / SAMPLE_RATE
            ax.plot(t, signal_snapshot[-display_samples:], linewidth=0.5, color='green')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Time Domain (RX Signal) [Zoom: {anim_state['zoom_duration']:.3f}s]")
        ax.grid(True, alpha=0.3)

        # Plot 2: Frequency domain (FFT with windowing)
        ax = axes[0, 1]
        ax.clear()
        fft_window = min(SAMPLE_RATE, len(signal_snapshot))
        if fft_window >= 4096:
            windowed_signal = signal_snapshot[-fft_window:] * np.hanning(fft_window)
            fft = np.abs(np.fft.fft(windowed_signal))
            freqs = np.fft.fftfreq(len(fft), 1 / SAMPLE_RATE)
            ax.plot(freqs[:len(freqs)//2], fft[:len(fft)//2], linewidth=0.5, color='green')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Frequency Domain (FFT)")
        ax.grid(True, alpha=0.3)

        # Plot 3: Bit comparison (Known TX vs Decoded RX)
        ax = axes[1, 0]
        ax.clear()

        all_decoded_bits = ''.join(rx_bits_snapshot)
        all_tx_bits = ''.join(tx_bits_snapshot)
        
        bits_to_show = min(20, len(all_decoded_bits), len(all_tx_bits))
        if bits_to_show > 0:
            tx_bits_short = np.array([int(b) for b in all_tx_bits[-bits_to_show:]])
            rx_bits_short = np.array([int(b) for b in all_decoded_bits[-bits_to_show:]])

            errors = (tx_bits_short != rx_bits_short)
            error_count = np.sum(errors)

            # Update statistics
            bits_compared = min(len(all_decoded_bits), len(all_tx_bits)) if len(all_tx_bits) > 0 else len(all_decoded_bits)
            if bits_compared > 0 and len(all_tx_bits) > 0:
                total_errors = np.sum(np.array([int(b) for b in all_tx_bits[:bits_compared]]) !=
                                     np.array([int(b) for b in all_decoded_bits[:bits_compared]]))
                current_ber = total_errors / bits_compared
            else:
                total_errors = 0
                current_ber = 0.0

            # Plot TX and RX bits
            ax.plot(range(bits_to_show), tx_bits_short, 'b-o', markersize=5, alpha=0.7, label='TX (known)', linewidth=1.5)
            ax.plot(range(bits_to_show), rx_bits_short, 'g-s', markersize=5, alpha=0.7, label='RX (decoded)', linewidth=1.5)

            # Highlight errors
            error_indices = np.where(errors)[0]
            if len(error_indices) > 0:
                ax.scatter(error_indices, tx_bits_short[error_indices],
                            color='red', s=150, marker='x', linewidths=3, label='Errors', zorder=5)

            ax.set_title(f'Bit Comparison (Last {bits_to_show} bits) - BER: {current_ber:.4f}')
            ax.set_xlabel('Bit Index')
            ax.set_ylabel('Bit Value')
            ax.set_ylim(-0.2, 1.2)
            ax.set_yticks([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=9)

            # Add error count annotation
            ax.text(0.02, 0.98, f'Errors (last {bits_to_show}): {error_count}/{bits_to_show}\nTotal: {total_errors}/{bits_compared}',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontsize=9)
        else:
            ax.text(0.5, 0.5, "Waiting for bits...", ha='center', va='center', fontsize=12)
            ax.set_title("Bit Comparison (Awaiting Signal)")


        # Plot 4: Constellation
        ax = axes[1, 1]
        if len(symbol_snapshot) > 0:
            ax.scatter(np.real(symbol_snapshot), np.imag(symbol_snapshot), alpha=0.6, s=50, color='green')
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlabel("I (In-phase)")
            ax.set_ylabel("Q (Quadrature)")
            ax.set_title(f"Constellation ({len(symbol_snapshot)} symbols)")
            ax.grid(True, alpha=0.3)
            max_lim = 1.5
            ax.set_xlim(-max_lim, max_lim)
            ax.set_ylim(-max_lim, max_lim)

        # Add status overlay (pause/zoom info)
        status_text = f"Zoom: {anim_state['zoom_duration']:.3f}s"
        if anim_state['paused']:
            status_text = "⏸ PAUSED | " + status_text
        fig.text(0.02, 0.98, status_text, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        fig.text(0.02, 0.93, "[SPACE] Pause | [+/-] Zoom | [0] Reset", transform=fig.transFigure, fontsize=9,
                verticalalignment='top', color='gray')

        return axes.flat

    # Create animation
    def on_close(_event):
        stop_event.set()

    def on_key_press(event):
        """Handle keyboard pause/zoom controls."""
        if event.key == ' ':
            # Toggle pause
            anim_state['paused'] = not anim_state['paused']
        elif event.key == '+' or event.key == '=':
            # Zoom in (decrease time window)
            anim_state['zoom_duration'] = max(0.001, anim_state['zoom_duration'] * 0.8)
        elif event.key == '-' or event.key == '_':
            # Zoom out (increase time window)
            anim_state['zoom_duration'] = min(5.0, anim_state['zoom_duration'] * 1.25)
        elif event.key == '0':
            # Reset to default
            anim_state['zoom_duration'] = 1.0

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    def playback_callback(outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        buffer = None
        with playback_lock:
            if playback_queue:
                buffer = playback_queue.popleft()

        if buffer is None:
            outdata[:, 0] = 0.0
            return

        if len(buffer) < frames:
            outdata[:len(buffer), 0] = buffer
            outdata[len(buffer):, 0] = 0.0
        else:
            outdata[:, 0] = buffer[:frames]
            remainder = buffer[frames:]
            if len(remainder) > 0:
                with playback_lock:
                    playback_queue.appendleft(remainder)

    def play_worker():
        # Start output stream for live playback of received audio
        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                device=audio_device,
                dtype='float32',
                callback=playback_callback,
                blocksize=4096,
            ):
                while not stop_event.is_set():
                    time.sleep(0.05)
        except Exception as e:
            print("Playback stream error:", e)

    player = threading.Thread(target=play_worker, daemon=True)
    player.start()

    worker = threading.Thread(target=rx_worker, daemon=True)
    worker.start()

    anim = FuncAnimation(fig, update_plots, interval=PLOT_UPDATE_INTERVAL_MS, blit=True)
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)

    if block:
        try:
            plt.show(block=True)
        finally:
            stop_event.set()
            worker.join(timeout=1.0)
    else:
        return {
            'stop_event': stop_event,
            'thread': worker,
            'fig': fig,
            'anim': anim,
        }

# ============================================================================
# VIRTUAL LOOPBACK DEVICE (for single-laptop testing)
# ============================================================================

# Simple in-process virtual audio device for loopback testing.
class VirtualLoopbackDevice:
    """Provide Output/Input context managers that transfer audio via an internal queue.

    Usage: pass the VirtualLoopbackDevice instance as `audio_device` to
    `run_transmitter_node` and `run_receiver_node`. The transmitter will open an
    output using `open_output`, and the receiver will open an input using
    `open_input` — both operate on the same internal buffer.
    """
    def __init__(self):
        from collections import deque
        self._queue = deque()
        self._cond = threading.Condition()
        self._stopped = False

    class _OutputCM:
        def __init__(self, parent):
            self._parent = parent

        def __enter__(self):
            return self

        def write(self, data):
            # accept 1-D float32 array
            with self._parent._cond:
                self._parent._queue.append(np.asarray(data, dtype=np.float32))
                self._parent._cond.notify()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _InputCM:
        def __init__(self, parent, callback, blocksize):
            self._parent = parent
            self._callback = callback
            self._blocksize = blocksize
            self._thread = None

        def __enter__(self):
            def run():
                while not self._parent._stopped:
                    with self._parent._cond:
                        while not self._parent._queue and not self._parent._stopped:
                            self._parent._cond.wait(timeout=0.1)
                        if self._parent._stopped:
                            break
                        buf = self._parent._queue.popleft()
                    if buf is None:
                        time.sleep(0.01)
                        continue

                    # Call the callback with shape (frames, channels)
                    frames = len(buf)
                    try:
                        self._callback(np.reshape(buf, (frames, 1)), frames, None, None)
                    except Exception:
                        # swallow exceptions from callback to keep loop running
                        pass

            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def open_output(self, *, samplerate, channels, dtype, blocksize):
        return VirtualLoopbackDevice._OutputCM(self)

    def open_input(self, *, samplerate, channels, dtype, callback, blocksize):
        return VirtualLoopbackDevice._InputCM(self, callback, blocksize)

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

# ============================================================================
# MAIN LOOP
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Laptop OTA Transceiver")
    parser.add_argument("--mode", required=True, choices=["tx", "rx", "loopback", "list-devices"],
                       help="Run as transmitter, receiver, or list audio devices")
    parser.add_argument("--device", type=int, default=None,
                       help="Audio device index (tx/rx modes only)")

    args = parser.parse_args()

    if args.mode == "list-devices":
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']}")
            print(f"     Channels: in={device['max_input_channels']}, out={device['max_output_channels']}")
        sys.exit(0)
    elif args.mode == "tx":
        run_transmitter_node(audio_device=args.device)
    elif args.mode == "rx":
        run_receiver_node(audio_device=args.device)
    elif args.mode == "loopback":
        # Create an in-process virtual device and start both nodes non-blocking.
        vdev = VirtualLoopbackDevice()

        tx_state = run_transmitter_node(audio_device=vdev, block=False)
        rx_state = run_receiver_node(audio_device=vdev, block=False)

        # Attach an on-close handler to stop both
        def on_close(_):
            vdev.stop()
            if tx_state and 'stop_event' in tx_state:
                tx_state['stop_event'].set()
            if rx_state and 'stop_event' in rx_state:
                rx_state['stop_event'].set()

        # Connect close events from both figures to the same handler
        try:
            tx_fig = tx_state.get('fig') if tx_state else None
            rx_fig = rx_state.get('fig') if rx_state else None
            if tx_fig is not None:
                tx_fig.canvas.mpl_connect('close_event', on_close)
            if rx_fig is not None:
                rx_fig.canvas.mpl_connect('close_event', on_close)

            # Show all figures in main thread
            plt.show(block=True)
        finally:
            vdev.stop()
            if tx_state and 'stop_event' in tx_state:
                tx_state['stop_event'].set()
            if rx_state and 'stop_event' in rx_state:
                rx_state['stop_event'].set()
