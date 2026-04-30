import re

import numpy as np
import scipy.signal as signal
import sounddevice as sd
from abc import ABC, abstractmethod
from typing import Tuple


# ============================================================================
# MODULATION BASE CLASS
# ============================================================================

class Modulation(ABC):
    """Abstract base class for modulation schemes."""

    @abstractmethod
    def modulate(self, bits: str) -> np.ndarray:
        """
        Convert bit string to complex symbols.

        Args:
            bits: String of '0' and '1' characters

        Returns:
            Complex symbol array
        """
        pass

    @abstractmethod
    def demodulate(self, symbols: np.ndarray) -> str:
        """
        Convert complex symbols back to bit string.

        Args:
            symbols: Complex symbol array

        Returns:
            Bit string
        """
        pass

    @property
    @abstractmethod
    def constellation(self) -> np.ndarray:
        """Return ideal constellation points."""
        pass

    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        """Return number of bits encoded per symbol."""
        pass
    
class nQAMModulation(Modulation):
    def __init__(self, n: int):
        """
        Initialize n-QAM modulation scheme.

        Args:
            n: Number of constellation points (must be a perfect square)

        Raises:
            ValueError: If n is not a perfect square
        """
        if int(np.sqrt(n)) ** 2 != n:
            raise ValueError(f"n must be a perfect square for square QAM, got {n}")
        
        self.n = n
        self.m = int(np.sqrt(n))
        self._constellation = self._generate_constellation()

    def _generate_constellation(self) -> np.ndarray:
        real_parts = np.arange(-self.m + 1, self.m, 2)
        imag_parts = np.arange(-self.m + 1, self.m, 2)
        constellation = np.array([r + 1j * i for r in real_parts for i in imag_parts])
        constellation = constellation / np.sqrt((constellation * np.conj(constellation)).mean())  # Normalize average symbol energy
        return constellation

    def modulate(self, bits: str) -> np.ndarray:
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(f"Bit string length must be multiple of {self.bits_per_symbol}, got {len(bits)}")

        symbols = []
        for i in range(0, len(bits), self.bits_per_symbol):
            bit_chunk = bits[i:i+self.bits_per_symbol]
            symbol_idx = int(bit_chunk, 2)
            symbols.append(self._constellation[symbol_idx])

        return np.array(symbols)

    def demodulate(self, symbols: np.ndarray) -> str:
        bits = ""
        real_levels = sorted(np.unique(np.real(self._constellation)))
        imag_levels = sorted(np.unique(np.imag(self._constellation)))

        for symbol in symbols:
            i_idx = np.argmin(np.abs(symbol.real - real_levels))
            q_idx = np.argmin(np.abs(symbol.imag - imag_levels))
            symbol_idx = i_idx * self.m + q_idx
            bits += format(symbol_idx, f'0{self.bits_per_symbol}b')

        return bits

    @property
    def constellation(self) -> np.ndarray:
        return self._constellation

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.n))
    

# ============================================================================
# FILTERS
# ============================================================================

def rrc_filter(samples_per_symbol: int, rolloff: float = 0.5, span: int = 4) -> np.ndarray:
    """
    Generate Root Raised Cosine (RRC) filter coefficients.

    Args:
        samples_per_symbol: Samples per symbol period
        rolloff: Rolloff factor (0 to 1)
        span: Filter span in symbol periods

    Returns:
        RRC filter coefficients
    """
    N = samples_per_symbol * span
    t = np.arange(-N, N + 1) / samples_per_symbol

    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:  # t = 0
            h[i] = 1 - rolloff + (4 * rolloff / np.pi)
        elif abs(abs(ti) - 1/(4*rolloff)) < 1e-10:  # Special case
            h[i] = rolloff * (np.sqrt(2) - 1) / (2 * np.sqrt(2))
        else:
            numerator = np.sin(np.pi * ti * (1 - rolloff)) + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))
            denominator = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            h[i] = numerator / denominator

    # Normalize energy
    h = h / np.sqrt(np.sum(h ** 2))
    return h

# ============================================================================
# TRANSMITTERS
# ============================================================================

class Transmitter:
    """
    Base transmitter class
    """

    def __init__(self, 
                modulation: Modulation, 
                carrier_freq: float,
                sample_rate: float, 
                bit_rate: float, 
                rolloff: float = 0.5, 
                phi_1: callable = lambda t, f: np.cos(2 * np.pi * f * t),
                phi_2: callable = lambda t, f: np.sin(2 * np.pi * f * t)):
        """
        Initialize transmitter.

        Args:
            modulation: Modulation instance (e.g., QPSKModulation())
            carrier_freq: Carrier frequency in Hz
            sample_rate: Sample rate in Hz
            bit_rate: Bit rate in bits/second
            rolloff: RRC rolloff factor

        Raises:
            ValueError: If sample_rate/baud_rate is not an integer
        """
        self.modulation = modulation
        self.sample_rate = sample_rate
        self.baud_rate = round(bit_rate / modulation.bits_per_symbol)

        self.carrier_freq = carrier_freq
        self.samples_per_symbol = round(self.sample_rate / self.baud_rate)
        
        self.phi_1 = lambda t: phi_1(t, carrier_freq)
        self.phi_2 = lambda t: phi_2(t, carrier_freq)
        
        self.rolloff = rolloff
        self._rrc_coeffs = rrc_filter(self.samples_per_symbol, rolloff=rolloff)

    def bits_to_symbols(self, bits: str) -> np.ndarray:
        """Convert bits to modulation symbols."""
        return self.modulation.modulate(bits)

    def upsample_and_filter(self, symbols: np.ndarray) -> np.ndarray:
        """
        Upsample symbols and apply RRC filter.

        Args:
            symbols: Complex symbol array

        Returns:
            Upsampled and filtered complex baseband signal
        """
        tx_signal = np.zeros(len(symbols) * self.samples_per_symbol, dtype=complex)
        tx_signal[::self.samples_per_symbol] = symbols
        tx_signal = signal.convolve(tx_signal, self._rrc_coeffs, mode='same')

        return tx_signal

    def modulate_to_passband(self, baseband_signal: np.ndarray) -> np.ndarray:
        """
        Modulate baseband signal to carrier frequency.

        Args:
            baseband_signal: Complex baseband signal

        Returns:
            Real passband signal
        """
        t = np.arange(len(baseband_signal)) / self.sample_rate
        
        modulated = baseband_signal.real * self.phi_1(t) + baseband_signal.imag * self.phi_2(t)

        # Normalize to avoid clipping
        max_val = np.max(np.abs(modulated))
        if max_val > 0:
            modulated = modulated / max_val * 0.9

        return modulated.astype(np.float32)

    def transmit_bits(self, bits: str) -> np.ndarray:
        """
        Complete transmit chain: bits -> symbols -> baseband -> passband.

        Args:
            bits: Bit string to transmit

        Returns:
            Real passband signal ready for audio transmission
        """
        symbols = self.bits_to_symbols(bits)
        baseband = self.upsample_and_filter(symbols)
        passband = self.modulate_to_passband(baseband)
        return passband

class TomasTransmitter:
   
    def __init__(self, carrier_freq: float = 4_000, sample_rate: float = 48_000,
                 baud_rate: float = 25, amplitude: float = 0.5):
        
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.samples_per_symbol = round(self.sample_rate / self.baud_rate)
        self.amplitude = amplitude

        self.payload_sec = 1.5

        self.payload_symbols = int(self.baud_rate * self.payload_sec)

        self.rng = np.random.default_rng(12345)
        self.preamble_bits = self.rng.integers(0, 2, size=64, dtype=int)

        self.payload_pattern = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)


    def bits_to_iq(self, b0, b1):
        if b0 == 0 and b1 == 0:
            return 1, 1
        if b0 == 0 and b1 == 1:
            return -1, 1
        if b0 == 1 and b1 == 1:
            return -1, -1
        return 1, -1

    def bits_to_symbols(self, bits):
        symbols = []

        for k in range(0, len(bits), 2):
            I, Q = self.bits_to_iq(bits[k], bits[k + 1])
            symbols.append((I + 1j * Q) / np.sqrt(2))

        return np.array(symbols)


    def transmit_bits(self, payload_bits = None):
        
        if payload_bits is None:
            payload_bits = np.resize(self.payload_pattern, 2 * self.payload_symbols)
        else: 
            payload_bits = np.concatenate([self.preamble_bits, np.array(list(payload_bits))])

        symbols = self.bits_to_symbols(payload_bits)
        bb = np.repeat(symbols, self.baud_rate)

        t = np.arange(len(bb)) / self.sample_rate

        audio = np.real(bb * np.exp(1j * 2 * np.pi * self.carrier_freq * t))

        audio = audio / np.max(np.abs(audio))
        audio = self.amplitude * audio

        return audio.astype(np.float32)




# ============================================================================
# RECEIVERS
# ============================================================================
class EmmettReceiver: 
    def __init__(self, 
                 modulation: Modulation = None, 
                 rolloff: float = None, 
                 bit_rate: float = 50, 
                 carrier_freq: float = 4_000, 
                 sample_rate: float = 48_000):
                
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
       
        self.SPS = bit_rate / 2  
        self.samples_per_symbol = round(self.sample_rate / self.SPS)

        self.chunk_sec = 5.0
        self.N = int(self.sample_rate * self.chunk_sec)

        self.LPF_window = int(self.SPS / 4)
        self.min_RX_amp = 0.005

        self.payload_sec = 1.5
        self.payload_symbols = int(self.SPS * self.payload_sec)

        self.discard_payload_start = 2
        self.discard_payload_end = 4

        self.rng = np.random.default_rng(12345)
        self.preamble_bits = self.rng.integers(0, 2, size=64, dtype=int)
        self.preamble_symbols = len(self.preamble_bits) // 2

        self.payload_pattern = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)


    def moving_average(self, x, window):
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="same")


    def bits_to_iq(self, b0, b1):
        if b0 == 0 and b1 == 0:
            return 1, 1
        if b0 == 0 and b1 == 1:
            return -1, 1
        if b0 == 1 and b1 == 1:
            return -1, -1
        return 1, -1


    def bits_to_complex_symbols(self, bits):
        symbols = []

        for k in range(0, len(bits), 2):
            I, Q = self.bits_to_iq(bits[k], bits[k + 1])
            symbols.append((I + 1j * Q) / np.sqrt(2))

        return np.array(symbols)


    def iq_to_bits(I, Q):
        if I >= 0 and Q >= 0:
            return [0, 0]
        if I < 0 and Q >= 0:
            return [0, 1]
        if I < 0 and Q < 0:
            return [1, 1]
        return [1, 0]


    def decode_symbols(self, z):
        bits = []

        for val in z:
            bits.extend(self.iq_to_bits(np.real(val), np.imag(val)))

        return np.array(bits, dtype=int)


    def best_payload_alignment(self, bits_hat):
        best_errors = None
        best_expected = None
        best_shift = None

        for shift_symbols in range(4):
            shift_bits = 2 * shift_symbols
            pattern_shifted = np.roll(self.payload_pattern, -shift_bits)
            expected = np.resize(pattern_shifted, len(bits_hat))

            errors = np.sum(bits_hat != expected)

            if best_errors is None or errors < best_errors:
                best_errors = errors
                best_expected = expected
                best_shift = shift_symbols

        return best_shift, best_errors, best_expected


    def find_preamble(self, z):
        known = self.bits_to_complex_symbols(self.preamble_bits)
        best = None

        n_known = np.arange(self.preamble_symbols)

        max_start = len(z) - self.preamble_symbols - self.payload_symbols

        if max_start <= 0:
            return None

        for start in range(max_start):
            test = z[start:start + self.preamble_symbols]

            phase_err = np.angle(test * np.conj(known))
            phase_err = np.unwrap(phase_err)

            phase_slope, phase0 = np.polyfit(n_known, phase_err, 1)

            correction = np.exp(-1j * (phase0 + phase_slope * n_known))
            test_corr = test * correction

            bits_hat = self.decode_symbols(test_corr)
            errors = np.sum(bits_hat != self.preamble_bits)

            if best is None or errors < best["errors"]:
                best = {
                    "start": start,
                    "phase0": phase0,
                    "phase_slope": phase_slope,
                    "errors": errors,
                }

        return best

    def get_phase_estimate(self):
        # Added for compatibility with StreamingTransceiver
        return self.phase_track if hasattr(self, 'phase_track') else 0.0 

    def receive_bits(self, bits=None):
        
        if bits is None:
            print("Recording...")

            r = sd.rec(self.N, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            r = r[:, 0]
        else:
            r = bits 

        rx_amp = np.max(np.abs(r))
        print("RX max amplitude:", rx_amp)

        if rx_amp < self.min_RX_amp:
            print("Signal too weak.")
            return None

        n = np.arange(len(r))
        t = n / self.sample_rate

        I_raw = 2 * r * np.cos(2 * np.pi * self.carrier_freq * t)
        Q_raw = -2 * r * np.sin(2 * np.pi * self.carrier_freq * t)

        I_filt = self.moving_average(I_raw, self.LPF_window)
        Q_filt = self.moving_average(Q_raw, self.LPF_window)

        best = None

        for offset in range(0, self.samples_per_symbol, max(1, self.samples_per_symbol // 120)):
            idx = np.arange(offset, len(r), self.samples_per_symbol)

            z = I_filt[idx] + 1j * Q_filt[idx]

            if len(z) < self.preamble_symbols + self.payload_symbols:
                continue

            cand = self.find_preamble(z)

            if cand is None:
                continue

            if best is None or cand["errors"] < best["errors"]:
                best = cand
                best["offset"] = offset
                best["z"] = z

        if best is None:
            print("Could not find preamble.")
            return None

        print("Best timing offset:", best["offset"])
        print("Preamble start symbol:", best["start"])
        print("Estimated phase0:", best["phase0"])
        print("Estimated phase slope:", best["phase_slope"])
        print("Preamble errors:", best["errors"], "/", len(self.preamble_bits))
        if best["errors"] > 2:
            print("Preamble lock not clean enough; skipping chunk.")
            return None

        n_sym = np.arange(len(best["z"])) - best["start"]
        self.phase_track = best["phase0"] + best["phase_slope"] * n_sym

        z_corr = best["z"] * np.exp(-1j * self.phase_track)

        payload_start = best["start"] + self.preamble_symbols
        payload_z = z_corr[payload_start:payload_start + self.payload_symbols]

        payload_z = payload_z[self.discard_payload_start:-self.discard_payload_end]

        if len(payload_z) == 0:
            print("No payload symbols.")
            return None

        bits_hat = self.decode_symbols(payload_z)

        shift, errors, bits_expected = self.best_payload_alignment(bits_hat)

        print("Best payload symbol shift:", shift)

        metrics = {
            "z_payload": payload_z,
            "bits_hat": bits_hat,
            "bits_expected": bits_expected,
            "errors": errors,
            "num_bits": len(bits_hat),
        }

        # Convert bits_hat from numpy array to bit string for StreamingTransceiver compatibility
        bits_string = ''.join(map(str, bits_hat))
        return (bits_string, best["z"])


class Receiver:
    """
    Base receiver class handling common signal processing.

    Uses a pluggable Modulation instance for demodulation-specific behavior.
    """

    def __init__(self, 
                 modulation: Modulation, 
                 carrier_freq: float,
                 sample_rate: float, 
                 bit_rate: float, 
                 rolloff: float = 0.5,
                 phi_1: callable = lambda t, f: np.cos(2 * np.pi * f * t),
                 phi_2: callable = lambda t, f: np.sin(2 * np.pi * f * t)):
        """
        Initialize receiver.

        Args:
            modulation: Modulation instance (e.g., QPSKModulation())
            carrier_freq: Carrier frequency in Hz
            sample_rate: Sample rate in Hz
            bit_rate: Bit rate in bits/second
            rolloff: RRC rolloff factor

        Raises:
            ValueError: If sample_rate/baud_rate is not an integer
        """
        self.modulation = modulation
        self.sample_rate = sample_rate
        self.baud_rate = round(bit_rate / modulation.bits_per_symbol)
        self.carrier_freq = carrier_freq
        self.samples_per_symbol = round(self.sample_rate / self.baud_rate)
        
        self.phi_1 = lambda t: phi_1(t, carrier_freq)
        self.phi_2 = lambda t: phi_2(t, carrier_freq)
        
        self.rolloff = rolloff
        self._rrc_coeffs = rrc_filter(self.samples_per_symbol, rolloff=rolloff)

        # Phase Lock Loop state
        self._phase_estimate = 0.0  # Estimated phase offset (radians)
        self._pll_integral = 0.0  # Integral term for PI controller
        self._pll_kp = 0.05  # Proportional gain
        self._pll_ki = 0.001  # Integral gain

    def _estimate_phase_error(self, symbols: np.ndarray) -> float:
        """
        Estimate phase error from received symbols using PLL error signal.
        Uses constellation closest-point detection to find phase misalignment.
        Averages over multiple symbols for robustness.

        Args:
            symbols: Received symbols

        Returns:
            Phase error estimate (radians)
        """
        if len(symbols) == 0:
            return 0.0

        num_symbols_to_use = min(10, len(symbols))
        symbols_to_process = symbols[-num_symbols_to_use:]

        phase_errors = []
        constellation = self.modulation.constellation

        for symbol in symbols_to_process:
            distances = np.abs(symbol - constellation)
            nearest_idx = np.argmin(distances)
            ideal_symbol = constellation[nearest_idx]

            if np.abs(np.min(distances)) > 1e-10:
                phase_error = np.angle(ideal_symbol) - np.angle(symbol)

                phase_error = np.angle(np.exp(1j * phase_error))
                phase_errors.append(phase_error)

        return np.mean(phase_errors) if phase_errors else 0.0

    def _update_pll(self, phase_error: float):
        """
        Update Phase Lock Loop with PI controller.

        Args:
            phase_error: Current phase error estimate
        """
        # PI controller
        self._pll_integral += self._pll_ki * phase_error
        self._phase_estimate = self._pll_kp * phase_error + self._pll_integral

        # Clamp to [-2*pi, 2*pi]
        while self._phase_estimate > 2 * np.pi:
            self._phase_estimate -= 2 * np.pi
        while self._phase_estimate < -2 * np.pi:
            self._phase_estimate += 2 * np.pi

    def demodulate_from_passband(self, passband_signal: np.ndarray,
                                 phase_offset: float = 0) -> np.ndarray:
        """
        Demodulate from carrier frequency to baseband.

        Args:
            passband_signal: Real passband signal
            phase_offset: Known phase offset to correct

        Returns:
            Complex baseband signal
        """
        # Remove DC offset
        passband_signal = passband_signal - np.mean(passband_signal)

        t = np.arange(len(passband_signal)) / self.sample_rate
       
        baseband = passband_signal * self.phi_1(t) + 1j * passband_signal * self.phi_2(t)
        phase_corrected = baseband * np.exp(-1j * phase_offset) / np.sqrt(2)

        return phase_corrected

    def filter_and_downsample(self, baseband_signal: np.ndarray) -> np.ndarray:
        """
        Apply matched filter (RRC) and downsample to symbol rate.

        Args:
            baseband_signal: Complex baseband signal

        Returns:
            Complex symbol array
        """
        baseband_filtered = signal.convolve(baseband_signal, self._rrc_coeffs, mode='same')

        symbols = []
        for i in range(0, len(baseband_filtered), self.samples_per_symbol):
            sample_idx = i
            if sample_idx < len(baseband_filtered):
                symbol = baseband_filtered[sample_idx]
                symbols.append(symbol)

        return np.array(symbols)

    def symbols_to_bits(self, symbols: np.ndarray) -> str:
        return self.modulation.demodulate(symbols)

    def receive_bits(self, passband_signal: np.ndarray,
                    phase_offset: float = 0) -> Tuple[str, np.ndarray]:
        """
        Complete receive chain with Phase Lock Loop: passband -> baseband -> symbols -> bits.

        Args:
            passband_signal: Real passband signal
            phase_offset: Additional known phase offset to correct

        Returns:
            (bit_string, symbol_array)
        """
        # Use cumulative PLL-estimated phase + any known offset
        total_phase_offset = self._phase_estimate + phase_offset

        baseband = self.demodulate_from_passband(passband_signal, total_phase_offset)
        symbols = self.filter_and_downsample(baseband)
        bits = self.symbols_to_bits(symbols)

        # Update PLL based on received symbols
        phase_error = self._estimate_phase_error(symbols)
        self._update_pll(phase_error)

        return bits, symbols

    def get_phase_estimate(self) -> float:
        return self._phase_estimate

    def reset_phase_lock(self):
        self._phase_estimate = 0.0
        self._pll_integral = 0.0
