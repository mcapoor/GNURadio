"""
Example: Adding 16-QAM Modulation

This file demonstrates how simple it is to add a new modulation scheme
using the framework. Just implement the Modulation interface!
"""

import numpy as np
from modulation_framework import Modulation, Transmitter, Receiver


class QAM16Modulation(Modulation):
    """16-QAM modulation: 4 bits per symbol."""

    # 4x4 grid of constellation points
    _constellation = np.array([
        3+3j, 3+1j, 3-1j, 3-3j,
        1+3j, 1+1j, 1-1j, 1-3j,
        -1+3j, -1+1j, -1-1j, -1-3j,
        -3+3j, -3+1j, -3-1j, -3-3j
    ]) / np.sqrt(10)  # Normalize for fair power comparison

    def modulate(self, bits: str) -> np.ndarray:
        """
        Map 4-bit words to 16 constellation points.

        1111 -> constellation[15] (-3-3j)
        0000 -> constellation[0] (3+3j)
        etc.
        """
        # Pad bits to multiple of 4
        if len(bits) % 4 != 0:
            bits += '0' * (4 - len(bits) % 4)

        symbols = []
        for i in range(0, len(bits), 4):
            nibble = bits[i:i+4]
            index = int(nibble, 2)  # Convert "0011" -> 3
            symbols.append(self._constellation[index])

        return np.array(symbols)

    def demodulate(self, symbols: np.ndarray) -> str:
        """
        Find nearest constellation point and map to 4-bit word.
        Uses nearest-neighbor hard detection.
        """
        bits = ""
        for symbol in symbols:
            distances = np.abs(symbol - self._constellation)
            nearest_idx = np.argmin(distances)
            bits += format(nearest_idx, '04b')  # Convert 3 -> "0011"

        return bits

    @property
    def constellation(self) -> np.ndarray:
        return self._constellation

    @property
    def bits_per_symbol(self) -> int:
        return 4


class BPSKModulation(Modulation):
    """BPSK (Binary PSK): 1 bit per symbol."""

    _constellation = np.array([1+0j, -1+0j])

    def modulate(self, bits: str) -> np.ndarray:
        return np.array([self._constellation[int(b)] for b in bits])

    def demodulate(self, symbols: np.ndarray) -> str:
        bits = ""
        for s in symbols:
            bits += "0" if s.real > 0 else "1"
        return bits

    @property
    def constellation(self) -> np.ndarray:
        return self._constellation

    @property
    def bits_per_symbol(self) -> int:
        return 1


def demo_qam16():
    """Demonstrate 16-QAM usage."""
    print("=" * 60)
    print("16-QAM Modulation Demo")
    print("=" * 60)

    # Create modulation instance
    qam16 = QAM16Modulation()

    # Create transmitter and receiver
    tx = Transmitter(
        modulation=qam16,
        carrier_freq=880,
        sample_rate=44100,
        baud_rate=100,
        rolloff=0.5
    )
    rx = Receiver(
        modulation=qam16,
        carrier_freq=880,
        sample_rate=44100,
        baud_rate=100,
        rolloff=0.5
    )

    # Test message
    bits = "00011011" * 10  # 80 bits = 20 symbols in 16-QAM
    print(f"\nOriginal bits: {bits[:40]}... ({len(bits)} bits)")
    print(f"Symbols encoding: {len(bits) // qam16.bits_per_symbol} symbols")
    print(f"Bits per symbol: {qam16.bits_per_symbol}")

    # Transmit
    tx_signal = tx.transmit_bits(bits)
    print(f"\nTransmitted signal: {len(tx_signal)} samples")

    # Receive (in ideal conditions - no noise)
    rx_bits, rx_symbols = rx.receive_bits(tx_signal)
    print(f"Received bits: {rx_bits[:40]}... ({len(rx_bits)} bits)")
    print(f"Received symbols: {len(rx_symbols)}")

    # Check modulation only (perfect recovery)
    symbols = qam16.modulate(bits)
    decoded = qam16.demodulate(symbols)
    print(f"\nModulation only (no DSP): {bits == decoded}")

    # Show constellation
    print(f"\n16-QAM Constellation (10 points shown):")
    for i, pt in enumerate(qam16.constellation[:10]):
        print(f"  {format(i, '04b')} -> {pt.real:+.3f}{pt.imag:+.3f}j")
    print(f"  ... ({len(qam16.constellation)} total points)")

    return qam16, tx, rx


def demo_bpsk():
    """Demonstrate BPSK usage."""
    print("\n" + "=" * 60)
    print("BPSK Modulation Demo")
    print("=" * 60)

    bpsk = BPSKModulation()

    tx = Transmitter(
        modulation=bpsk,
        carrier_freq=880,
        sample_rate=44100,
        baud_rate=100,
        rolloff=0.5
    )

    bits = "0101" * 20
    tx_signal = tx.transmit_bits(bits)

    print(f"\nBPSK constellation: {bpsk.constellation}")
    print(f"Bits encoded: {len(bits)} bits = {len(bits)} symbols (1 bit/symbol)")
    print(f"Transmitted signal: {len(tx_signal)} samples")


def compare_modulations():
    """Compare properties of different modulation schemes."""
    print("\n" + "=" * 60)
    print("Modulation Scheme Comparison")
    print("=" * 60)

    schemes = [
        BPSKModulation(),
        # QPSKModulation(),  # Uncomment to include
        QAM16Modulation(),
    ]

    print(f"\n{'Scheme':<15} {'Bits/Sym':<12} {'Const. Size':<15}")
    print("-" * 42)
    for mod in schemes:
        print(f"{mod.__class__.__name__:<15} {mod.bits_per_symbol:<12} {len(mod.constellation):<15}")


if __name__ == "__main__":
    qam16, tx_qam, rx_qam = demo_qam16()
    demo_bpsk()
    compare_modulations()

    print("\n" + "=" * 60)
    print("Note: Just implement Modulation interface to add new schemes!")
    print("The Transmitter/Receiver handle all DSP automatically.")
    print("=" * 60)
