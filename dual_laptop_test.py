"""
Dual-Laptop OTA Transceiver Test (No Networking)
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

from modulation_framework import Transmitter, Receiver, nQAMModulation, TomasTransmitter, EmmettReceiver
from simulation_framework import (
    channel_model,
    identity_channel_model,
    bit_stream_generator,
    StreamingTransceiver,
    create_animated_transceiver_plot,
)

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import argparse
import time
import threading

plt.ion()

# ============================================================================
# CONFIGURATION
# ============================================================================

CARRIER_FREQ = 880
SAMPLE_RATE = 44_000
BITS_PER_SECOND = 500
SNR = 20  # dB

CHUNK_SIZE = 50

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
    modulation = nQAMModulation(4)  # QPSK
    baud_rate = round(BITS_PER_SECOND / modulation.bits_per_symbol)

    # Create transmitter
    transmitter = Transmitter(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        baud_rate=baud_rate
    )

    # Dummy receiver (won't use)
    receiver = Receiver(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        baud_rate=baud_rate
    )

    def tx_channel(signal_in):
        """TX side channel - identity."""
        return signal_in

    # Create streaming transceiver (TX only)
    streamer = StreamingTransceiver(
        transmitter=transmitter,
        receiver=receiver,
        channel_model_fn=tx_channel,
        sample_rate=SAMPLE_RATE,
        snr_db=SNR,
        bits_per_symbol=modulation.bits_per_symbol,
        audio_device=audio_device
    )

    # Bit stream generator
    source = bit_stream_generator(0.5, chunk_size=CHUNK_SIZE)

    # TX loop: generate bits, transmit via speaker
    def tx_worker():
        print("Starting transmission loop...\n")
        try:
            chunk_num = 0
            while True:
                bit_chunk = next(source)

                # Process through transmitter
                tx_signal = transmitter.transmit_bits(bit_chunk)
                tx_symbols = transmitter.bits_to_symbols(bit_chunk)

                # Play audio on speaker
                sd.play(tx_signal, samplerate=SAMPLE_RATE, device=audio_device)

                # Update local buffers
                streamer.tx_bits_buffer += bit_chunk
                streamer.tx_symbols_buffer = np.append(streamer.tx_symbols_buffer, tx_symbols)
                streamer.tx_signal_buffer = np.append(streamer.tx_signal_buffer, tx_signal)

                # Trim buffers to prevent memory bloat
                if len(streamer.tx_symbols_buffer) > 5000:
                    trim_idx = len(streamer.tx_symbols_buffer) - 5000
                    streamer.tx_bits_buffer = streamer.tx_bits_buffer[trim_idx * 2:]
                    streamer.tx_symbols_buffer = streamer.tx_symbols_buffer[trim_idx:]
                    streamer.tx_signal_buffer = streamer.tx_signal_buffer[trim_idx * SAMPLE_RATE // baud_rate:]

                chunk_num += 1
                if chunk_num % 10 == 0:
                    print(f"TX Chunk {chunk_num}: sent {len(bit_chunk)} bits")

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nTransmitter stopped.")

    # Start TX worker thread
    tx_thread = threading.Thread(target=tx_worker, daemon=True)
    tx_thread.start()

    # No-op generator (real bits come from tx worker thread)
    def noop_gen():
        while True:
            yield ""

    # Create visualization
    fig, anim = create_animated_transceiver_plot(
        streamer,
        noop_gen(),
        update_interval_ms=500,
        max_frames=None
    )

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
    print("="*60 + "\n")

    # Setup modulation (same as TX)
    modulation = nQAMModulation(4)
    baud_rate = round(BITS_PER_SECOND / modulation.bits_per_symbol)

    # Create receiver
    receiver = Receiver(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        baud_rate=baud_rate
    )

    # Dummy transmitter (won't use)
    transmitter = Transmitter(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        baud_rate=baud_rate
    )

    def rx_channel(signal_in):
        """RX side - identity channel."""
        return signal_in

    streamer = StreamingTransceiver(
        transmitter=transmitter,
        receiver=receiver,
        channel_model_fn=rx_channel,
        sample_rate=SAMPLE_RATE,
        snr_db=SNR,
        bits_per_symbol=modulation.bits_per_symbol,
        audio_device=audio_device
    )

    # Recording thread
    def recording_worker():
        """Continuously record from mic and demodulate."""
        chunk_duration = 0.5  # seconds
        chunk_samples = int(SAMPLE_RATE * chunk_duration)

        print("Starting mic recording...\n")
        try:
            chunk_num = 0
            while True:
                # Record chunk from mic
                rec = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1, device=audio_device,
                            dtype='float32')
                sd.wait()
                rx_signal = rec[:, 0]

                # Demodulate
                result = receiver.receive_bits(rx_signal)
                if result:
                    rx_bits, rx_symbols = result
                    # Update streamer
                    streamer.rx_bits_buffer += rx_bits
                    streamer.rx_symbols_buffer = np.append(streamer.rx_symbols_buffer, rx_symbols)

                streamer.rx_signal_buffer = np.append(streamer.rx_signal_buffer, rx_signal)

                # Trim buffers
                if len(streamer.rx_symbols_buffer) > 5000:
                    trim_idx = len(streamer.rx_symbols_buffer) - 5000
                    streamer.rx_bits_buffer = streamer.rx_bits_buffer[trim_idx * 2:]
                    streamer.rx_symbols_buffer = streamer.rx_symbols_buffer[trim_idx:]
                    streamer.rx_signal_buffer = streamer.rx_signal_buffer[trim_idx * SAMPLE_RATE // baud_rate:]

                chunk_num += 1
                if chunk_num % 10 == 0:
                    print(f"RX Chunk {chunk_num}: received signal")

        except KeyboardInterrupt:
            print("\nRecording stopped.")

    rec_thread = threading.Thread(target=recording_worker, daemon=True)
    rec_thread.start()

    # No-op generator (real bits come from mic recording thread)
    def noop_gen():
        while True:
            yield ""

    # Create visualization
    fig, anim = create_animated_transceiver_plot(
        streamer,
        noop_gen(),
        update_interval_ms=500,
        max_frames=None
    )

    plt.show(block=True)


# ============================================================================
# LOOPBACK TEST (SINGLE LAPTOP)
# ============================================================================

def run_loopback_test():
    """
    Test TX and RX on the same laptop with audio loopback.
    TX generates audio that is immediately fed to RX (in-memory).
    Both are visualized together for full system validation.
    """
    print("\n" + "="*60)
    print("LOOPBACK TEST (Single Laptop)")
    print("="*60)
    print(f"Testing TX and RX with internal loopback")
    print(f"Bit rate: {BITS_PER_SECOND} bps")
    print("No audio hardware needed - perfect channel")
    print("="*60 + "\n")

    # Setup modulation
    modulation = nQAMModulation(4)  # QPSK
    baud_rate = round(BITS_PER_SECOND / modulation.bits_per_symbol)

    # Create transmitter
    transmitter = Transmitter(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        baud_rate=baud_rate
    )

    # Create receiver (same config)
    receiver = Receiver(
        modulation=modulation,
        carrier_freq=CARRIER_FREQ,
        sample_rate=SAMPLE_RATE,
        baud_rate=baud_rate
    )

    def loopback_channel(signal_in):
        """Loopback channel - identity (perfect channel)."""
        return signal_in

    # Create single streamer for loopback
    streamer = StreamingTransceiver(
        transmitter=transmitter,
        receiver=receiver,
        channel_model_fn=loopback_channel,
        sample_rate=SAMPLE_RATE,
        snr_db=SNR,
        bits_per_symbol=modulation.bits_per_symbol
    )

    # Bit stream generator
    bit_gen = bit_stream_generator(0.5, chunk_size=CHUNK_SIZE)

    # Loopback worker thread
    def loopback_worker():
        """Generate TX audio and immediately process as RX."""
        print("Starting loopback loop...\n")
        try:
            chunk_num = 0
            while True:
                bit_chunk = next(bit_gen)

                # TX: Generate audio
                tx_signal = transmitter.transmit_bits(bit_chunk)
                tx_symbols = transmitter.bits_to_symbols(bit_chunk)

                # RX: Process the same audio immediately
                result = receiver.receive_bits(tx_signal)
                if result:
                    rx_bits, rx_symbols = result
                else:
                    rx_bits = ""
                    rx_symbols = np.array([], dtype=complex)

                # Update TX and RX buffers
                streamer.tx_bits_buffer += bit_chunk
                streamer.tx_symbols_buffer = np.append(streamer.tx_symbols_buffer, tx_symbols)
                streamer.tx_signal_buffer = np.append(streamer.tx_signal_buffer, tx_signal)

                streamer.rx_bits_buffer += rx_bits
                streamer.rx_symbols_buffer = np.append(streamer.rx_symbols_buffer, rx_symbols)
                streamer.rx_signal_buffer = np.append(streamer.rx_signal_buffer, tx_signal)

                # Track phase
                phase_estimate = receiver.get_phase_estimate()
                streamer.phase_history.append(np.degrees(phase_estimate))

                # Update BER
                bits_to_compare = min(len(streamer.tx_bits_buffer), len(streamer.rx_bits_buffer))
                if bits_to_compare > 0:
                    tx_arr = np.array(list(streamer.tx_bits_buffer[:bits_to_compare]), dtype=int)
                    rx_arr = np.array(list(streamer.rx_bits_buffer[:bits_to_compare]), dtype=int)
                    errors = np.sum(tx_arr != rx_arr)
                    ber = errors / bits_to_compare
                    streamer.running_ber_history.append(ber)

                # Trim buffers to prevent memory bloat
                if len(streamer.tx_symbols_buffer) > 5000:
                    trim_idx = len(streamer.tx_symbols_buffer) - 5000
                    streamer.tx_bits_buffer = streamer.tx_bits_buffer[trim_idx * 2:]
                    streamer.rx_bits_buffer = streamer.rx_bits_buffer[trim_idx * 2:]
                    streamer.tx_symbols_buffer = streamer.tx_symbols_buffer[trim_idx:]
                    streamer.rx_symbols_buffer = streamer.rx_symbols_buffer[trim_idx:]
                    streamer.tx_signal_buffer = streamer.tx_signal_buffer[trim_idx * SAMPLE_RATE // baud_rate:]
                    streamer.rx_signal_buffer = streamer.rx_signal_buffer[trim_idx * SAMPLE_RATE // baud_rate:]

                chunk_num += 1
                if chunk_num % 10 == 0:
                    ber_str = f"{streamer.running_ber_history[-1]:.4f}" if streamer.running_ber_history else "N/A"
                    print(f"Loopback Chunk {chunk_num}: TX {len(bit_chunk)} bits, RX {len(rx_bits)} bits, BER={ber_str}")

                time.sleep(0.1)  # Loopback faster than real-time

        except KeyboardInterrupt:
            print("\nLoopback test stopped.")

    # Start loopback worker thread
    loopback_thread = threading.Thread(target=loopback_worker, daemon=True)
    loopback_thread.start()

    # No-op generator (real bits come from loopback worker thread)
    def noop_gen():
        while True:
            yield ""

    # Create visualization showing both TX and RX
    fig, anim = create_animated_transceiver_plot(
        streamer,
        noop_gen(),
        update_interval_ms=500,
        max_frames=None
    )

    plt.show(block=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Laptop OTA Transceiver")
    parser.add_argument("--mode", required=True, choices=["tx", "rx", "loopback"],
                       help="Run as transmitter, receiver, or single-laptop loopback test")
    parser.add_argument("--device", type=int, default=None,
                       help="Audio device index (tx/rx modes only)")

    args = parser.parse_args()

    if args.mode == "tx":
        run_transmitter_node(audio_device=args.device)
    elif args.mode == "rx":
        run_receiver_node(audio_device=args.device)
    else:
        run_loopback_test()
