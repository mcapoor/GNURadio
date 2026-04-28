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

SINGLE LAPTOP (Loopback test - no audio hardware needed):
  $ python dual_laptop_test.py --mode loopback
"""

from modulation_framework import Transmitter, Receiver, nQAMModulation

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import time
import sys

plt.ion()

# ============================================================================
# CONFIGURATION
# ============================================================================

CARRIER_FREQ = 10_000
SAMPLE_RATE = 44_100
BITS_PER_SECOND = 10_000

CHUNK_SIZE = 50

CONSTELLATION_POINTS = 4

RNG_SEED = 12345

# Receiver chunk duration (in seconds) - increase for better demodulation
RX_CHUNK_DURATION = 0.5  # 0.5 seconds gives ~5 symbols at 10 baud rate

# ============================================================================
# TRANSMITTER NODE
# ============================================================================

def run_transmitter_node(audio_device=None):
    """Run transmitter that streams audio via speaker and displays local graphs."""
    print("\n" + "="*60)
    print("TRANSMITTER NODE")
    print("="*60)
    print(f"Transmitting via speaker at {CARRIER_FREQ} Hz carrier")
    print(f"Bit rate: {BITS_PER_SECOND} bps")
    print("="*60 + "\n")

    # Setup modulation
    modulation = nQAMModulation(CONSTELLATION_POINTS)

    transmitter = Transmitter(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        bit_rate=BITS_PER_SECOND    
    )

    bit_list = []
    symbol_list = []
    signal_list = []

    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Transmitter Live View", fontsize=16, fontweight='bold')

    bit_chunk = ''.join(str(x) for x in rng.integers(0, 2, size=CHUNK_SIZE, dtype=int))
    signal = transmitter.transmit_bits(bit_chunk)
    symbols = transmitter.bits_to_symbols(bit_chunk)
    
    def update_plots(frame):
        """Generate one chunk and update plots."""
        nonlocal bit_list, symbol_list, signal_list

        # Generate one chunk of bits
        bit_list.append(bit_chunk)
        symbol_list.extend(symbols)
        signal_list.extend(signal)

        # Keep only last ~2 seconds of signal 
        max_signal_samples = SAMPLE_RATE * 2
        if len(signal_list) > max_signal_samples:
            trim_amount = len(signal_list) - max_signal_samples
            signal_list[:] = signal_list[trim_amount:]

        # Keep last 200 bits for display
        if len(''.join(bit_list)) > 200:
            bit_list[:] = (bit_list[:-1] if len(bit_list[-1]) > 1 else bit_list[:-2])

        # Keep last 500 symbols
        if len(symbol_list) > 500:
            symbol_list[:] = symbol_list[-500:]

        # Plot 1: Time-domain signal
        ax = axes[0, 0]
        ax.clear()
        display_samples = min(SAMPLE_RATE, len(signal_list))
        if display_samples > 0:
            t = np.arange(display_samples) / SAMPLE_RATE
            ax.plot(t, signal_list[-display_samples:], linewidth=0.5, color='blue')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain (TX Signal)")
        ax.grid(True, alpha=0.3)

        # Plot 2: Frequency domain (FFT with windowing)
        ax = axes[0, 1]
        ax.clear()
        if len(signal_list) >= SAMPLE_RATE:
            windowed_signal = signal_list[-SAMPLE_RATE:] * np.hanning(SAMPLE_RATE)
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
        all_bits = ''.join(bit_list)
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
        if len(symbol_list) > 0:
            ax.scatter(np.real(symbol_list), np.imag(symbol_list), alpha=0.6, s=50, color='blue')
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlabel("I (In-phase)")
            ax.set_ylabel("Q (Quadrature)")
            ax.set_title(f"Constellation ({len(symbol_list)} symbols)")
            ax.grid(True, alpha=0.3)
            max_lim = 1.5
            ax.set_xlim(-max_lim, max_lim)
            ax.set_ylim(-max_lim, max_lim)

        return axes.flat

    # Create animation
    sd.play(signal, samplerate=SAMPLE_RATE, device=audio_device)

    anim = FuncAnimation(fig, update_plots, interval=200, blit=False)
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.show(block=True)


# ============================================================================
# RECEIVER NODE
# ============================================================================

def run_receiver_node(audio_device=None):
    """Run receiver that records from mic and displays local graphs."""
    print("\n" + "="*60)
    print("RECEIVER NODE")
    print("="*60)
    print(f"Listening on mic at {CARRIER_FREQ} Hz carrier")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"RNG Seed (for known TX pattern): {RNG_SEED}")
    print("="*60 + "\n")

    # Setup modulation (same as TX)
    modulation = nQAMModulation(CONSTELLATION_POINTS)

    receiver = Receiver(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        bit_rate=BITS_PER_SECOND
    )

    bit_list = []
    symbol_list = []
    signal_list = []
    chunk_count = 0

    rng = np.random.default_rng(RNG_SEED)
    known_tx_bits = ""

    total_bits_compared = 0
    total_bit_errors = 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Receiver Live View", fontsize=16, fontweight='bold')

    chunk_samples = int(SAMPLE_RATE * RX_CHUNK_DURATION)

    rec = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1, device=audio_device, dtype='float32')
    sd.wait()
    rx_signal = rec[:, 0] if rec.ndim > 1 else rec

    def update_plots(frame):
        """Record one chunk and update plots."""
        nonlocal bit_list, symbol_list, signal_list, chunk_count, known_tx_bits, total_bits_compared, total_bit_errors

        # Demodulate
        result = receiver.receive_bits(rx_signal)
        if result:
            rx_bits, rx_symbols = result
            bit_list.append(rx_bits)
            symbol_list.extend(rx_symbols)

        # Generate more known TX bits if needed (to match what was decoded)
        # This simulates the TX sequence using the same RNG seed
        all_decoded_bits = ''.join(bit_list)
        while len(known_tx_bits) < len(all_decoded_bits) + CHUNK_SIZE:
            tx_chunk = ''.join(str(x) for x in rng.integers(0, 2, size=CHUNK_SIZE, dtype=int))
            known_tx_bits += tx_chunk

        # Append raw signal
        signal_list.extend(rx_signal)

        # Keep only last ~2 seconds of signal
        max_signal_samples = SAMPLE_RATE * 2
        if len(signal_list) > max_signal_samples:
            trim_amount = len(signal_list) - max_signal_samples
            signal_list[:] = signal_list[trim_amount:]

        # Keep last 500 symbols
        if len(symbol_list) > 500:
            symbol_list[:] = symbol_list[-500:]

        chunk_count += 1

        # Plot 1: Time-domain signal
        ax = axes[0, 0]
        ax.clear()
        display_samples = min(SAMPLE_RATE, len(signal_list))
        if display_samples > 0:
            t = np.arange(display_samples) / SAMPLE_RATE
            ax.plot(t, signal_list[-display_samples:], linewidth=0.5, color='green')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain (RX Signal)")
        ax.grid(True, alpha=0.3)

        # Plot 2: Frequency domain (FFT with windowing)
        ax = axes[0, 1]
        ax.clear()
        if len(signal_list) >= SAMPLE_RATE:
            windowed_signal = signal_list[-SAMPLE_RATE:] * np.hanning(SAMPLE_RATE)
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

        all_decoded_bits = ''.join(bit_list)
        bits_to_show = min(20, len(all_decoded_bits), len(known_tx_bits))

        if bits_to_show > 0:
            tx_bits_short = np.array([int(b) for b in known_tx_bits[-bits_to_show:]])
            rx_bits_short = np.array([int(b) for b in all_decoded_bits[-bits_to_show:]])

            # Calculate errors for this window
            errors = (tx_bits_short != rx_bits_short)
            error_count = np.sum(errors)

            # Update statistics
            bits_compared = min(len(all_decoded_bits), len(known_tx_bits))
            if bits_compared > 0:
                total_errors = np.sum(np.array([int(b) for b in known_tx_bits[:bits_compared]]) !=
                                     np.array([int(b) for b in all_decoded_bits[:bits_compared]]))
                current_ber = total_errors / bits_compared
            else:
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
            ax.text(0.02, 0.98, f'Errors (last 20): {error_count}/{bits_to_show}\nTotal: {total_errors}/{bits_compared}',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontsize=9)
        else:
            ax.text(0.5, 0.5, "Waiting for bits...", ha='center', va='center', fontsize=12)
            ax.set_title("Bit Comparison (Awaiting Signal)")


        # Plot 4: Constellation
        ax = axes[1, 1]
        ax.clear()
        if len(symbol_list) > 0:
            ax.scatter(np.real(symbol_list), np.imag(symbol_list), alpha=0.6, s=50, color='green')
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlabel("I (In-phase)")
            ax.set_ylabel("Q (Quadrature)")
            ax.set_title(f"Constellation ({len(symbol_list)} symbols)")
            ax.grid(True, alpha=0.3)
            max_lim = 1.5
            ax.set_xlim(-max_lim, max_lim)
            ax.set_ylim(-max_lim, max_lim)

        return axes.flat

    # Create animation
    anim = FuncAnimation(fig, update_plots, interval=200, blit=False)
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.show(block=True)


# ============================================================================
# MAIN LOOP
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Laptop OTA Transceiver")
    parser.add_argument("--mode", required=True, choices=["tx", "rx", "list-devices"],
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
