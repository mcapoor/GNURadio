import numpy as np
import scipy.signal as signal
from scipy.linalg import hadamard 

from abc import ABC, abstractmethod
from typing import Tuple
from itertools import cycle
from typing import Union, Tuple

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

    #TODO: added this; ensure compatible w all else
    @abstractmethod
    def to_passband(self, symbols: np.ndarray, carrier_freq: float, 
                    sample_rate: float, samples_per_symbol: int) -> np.ndarray:
        """
        Convert baseband symbols into a real-world audio waveform.
        Each scheme (QPSK, FSK, etc.) implements its own physics here.
        """
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
    
    def _get_rrc_coeffs(self, sps: int, alpha: float, span: int = 10) -> np.ndarray:
        """Helper that wraps the partner's filter logic."""
        return rrc_filter(samples_per_symbol=sps, rolloff=alpha, span=span)

    def to_passband(self, symbols: np.ndarray, carrier_freq: float, 
                    sample_rate: float, samples_per_symbol: int,
                    filter_type: str = "rrc", rolloff: float = 0.5) -> np.ndarray:
        """
        Standard physics engine for Linear Modulation (QPSK, QAM).
        Converts symbols to real-world audio samples.
        """
        # 1. Create Impulse Train (Spikes separated by zeros)
        num_samples = len(symbols) * samples_per_symbol
        upsampled_I = np.zeros(num_samples)
        upsampled_Q = np.zeros(num_samples)
        
        upsampled_I[::samples_per_symbol] = np.real(symbols)
        upsampled_Q[::samples_per_symbol] = np.imag(symbols)
        
        # 2. Apply Shaping Filter
        if filter_type == "rect":
            # Simulate partner's old np.repeat logic using convolution
            rect_pulse = np.ones(samples_per_symbol)
            I_base = signal.convolve(upsampled_I, rect_pulse, mode='same')
            Q_base = signal.convolve(upsampled_Q, rect_pulse, mode='same')
        else:
            # Use the professional RRC filter
            rrc = self._get_rrc_coeffs(samples_per_symbol, rolloff, span=10)
            # mode='same' ensures output length matches upsampled_I length exactly
            I_base = signal.convolve(upsampled_I, rrc, mode='same')
            Q_base = signal.convolve(upsampled_Q, rrc, mode='same')

        # 3. Carrier Modulation
        # Generate time axis matching the baseband length
        t = np.arange(len(I_base)) / sample_rate
        
        # Standard I/Q Modulation formula
        passband = I_base * np.cos(2 * np.pi * carrier_freq * t) - \
                Q_base * np.sin(2 * np.pi * carrier_freq * t)
                
        return passband.astype(np.float32)
    
class PSKModulation(Modulation):
    def __init__(self, f0_offset: float, f1_offset: float, fs: float, spb: int):
        """
        f0_offset: Hz relative to carrier (e.g., -500)
        f1_offset: Hz relative to carrier (e.g., +500)
        """
        self.f0 = f0_offset
        self.f1 = f1_offset
        self.fs = fs
        self.spb = spb

    def modulate(self, bits: str) -> np.ndarray:
        """
        In this modular framework, we return 'symbols'. 
        For FSK, each symbol is a chunk of complex exponential.
        """
        n = np.arange(self.spb)
        audio_chunks = []
        
        for bit in bits:
            freq = self.f1 if bit == '1' else self.f0
            # Generate a complex phasor for this bit
            chunk = np.exp(1j * 2 * np.pi * freq * n / self.fs)
            audio_chunks.append(chunk)
            
        return np.concatenate(audio_chunks)

    def demodulate(self, complex_baseband: np.ndarray) -> str:
        bits = ""
        # Process in chunks of Samples Per Bit
        for i in range(0, len(complex_baseband), self.spb):
            chunk = complex_baseband[i:i + self.spb]
            if len(chunk) < self.spb: break
            
            # Non-coherent detection: Which frequency is stronger?
            e0 = self._get_energy(chunk, self.f0)
            e1 = self._get_energy(chunk, self.f1)
            
            bits += "1" if e1 > e0 else "0"
        return bits

    def _get_energy(self, chunk, freq_offset):
        n = np.arange(len(chunk))
        # Mix the complex signal down by the offset and sum
        ref = np.exp(-1j * 2 * np.pi * freq_offset * n / self.fs)
        return np.abs(np.sum(chunk * ref))**2

    @property
    def constellation(self):
        return np.array([1+0j]) # FSK doesn't use a standard IQ constellation

    @property
    def bits_per_symbol(self):
        return 1
    
    def to_passband(self, symbols: np.ndarray, carrier_freq: float, 
                    sample_rate: float, samples_per_symbol: int, **kwargs) -> np.ndarray:
        """
        FSK to_passband: Ignores I/Q math and generates frequency tones.
        Note: 'symbols' from BFSKModulation.modulate are already complex phasors.
        """
        # In your BFSK modulate, you already generated the complex baseband.
        # We just need to shift it to the carrier_freq.
        t = np.arange(len(symbols)) / sample_rate
        
        # Real-part of (Baseband * Carrier)
        # This converts your complex frequency offsets into real passband audio
        return np.real(symbols * np.exp(1j * 2 * np.pi * carrier_freq * t))

class GrayCodedQPSK(Modulation):
    @property
    def bits_per_symbol(self) -> int:
        return 2

    @property
    def constellation(self) -> np.ndarray:
        return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)

    def modulate(self, bits: str) -> np.ndarray:
        if len(bits) % 2 != 0:
            raise ValueError("QPSK requires an even number of bits.")
        
        mapping = {"00": 1+1j, "01": -1+1j, "11": -1-1j, "10": 1-1j}
        return np.array([mapping[bits[i:i+2]] / np.sqrt(2) for i in range(0, len(bits), 2)])

    def demodulate(self, symbols: np.ndarray) -> str:
        res = []
        for z in symbols:
            I, Q = np.real(z), np.imag(z)
            if I >= 0 and Q >= 0: res.append("00")
            elif I < 0 and Q >= 0: res.append("01")
            elif I < 0 and Q < 0: res.append("11")
            else: res.append("10")
        return "".join(res)

    def _get_rrc_coeffs(self, sps: int, alpha: float, span: int = 10) -> np.ndarray:
        """Helper that wraps the partner's filter logic."""
        return rrc_filter(samples_per_symbol=sps, rolloff=alpha, span=span)

    def to_passband(self, symbols: np.ndarray, carrier_freq: float, 
                    sample_rate: float, samples_per_symbol: int,
                    filter_type: str = "rrc", rolloff: float = 0.5) -> np.ndarray:
        """
        Standard physics engine for Linear Modulation (QPSK, QAM).
        Converts symbols to real-world audio samples.
        """
        # 1. Create Impulse Train (Spikes separated by zeros)
        num_samples = len(symbols) * samples_per_symbol
        upsampled_I = np.zeros(num_samples)
        upsampled_Q = np.zeros(num_samples)
        
        upsampled_I[::samples_per_symbol] = np.real(symbols)
        upsampled_Q[::samples_per_symbol] = np.imag(symbols)
        
        # 2. Apply Shaping Filter
        if filter_type == "rect":
            # Simulate partner's old np.repeat logic using convolution
            rect_pulse = np.ones(samples_per_symbol)
            I_base = signal.convolve(upsampled_I, rect_pulse, mode='same')
            Q_base = signal.convolve(upsampled_Q, rect_pulse, mode='same')
        else:
            # Use the professional RRC filter
            rrc = self._get_rrc_coeffs(samples_per_symbol, rolloff, span=10)
            # mode='same' ensures output length matches upsampled_I length exactly
            I_base = signal.convolve(upsampled_I, rrc, mode='same')
            Q_base = signal.convolve(upsampled_Q, rrc, mode='same')

        # 3. Carrier Modulation
        # Generate time axis matching the baseband length
        t = np.arange(len(I_base)) / sample_rate
        
        # Standard I/Q Modulation formula
        passband = I_base * np.cos(2 * np.pi * carrier_freq * t) - \
                Q_base * np.sin(2 * np.pi * carrier_freq * t)
                
        return passband.astype(np.float32)
    
class CDMAModulation(Modulation):
    def __init__(self, n_users: int, spreading_factor: int):
        if not (spreading_factor > 0 and (spreading_factor & (spreading_factor - 1) == 0)):
            raise ValueError("Spreading factor must be a power of 2 for Walsh codes.")
        
        self.n_users = n_users
        self.sf = spreading_factor
        # Generate Walsh-Hadamard codes (each row is a unique user code)
        self.codes = hadamard(self.sf) 
        self._constellation = np.array([-1, 1]) # BPSK-like chips

    @property
    def bits_per_symbol(self) -> int:
        # In CDMA, one 'symbol' is actually one bit spread across SF chips
        return 1 

    @property
    def constellation(self) -> np.ndarray:
        return self._constellation

    def modulate(self, bits_string: str) -> np.ndarray:
        """
        Expects a string of bits for a SINGLE user, or 
        multiple users concatenated if handled by the caller.
        Returns BPSK symbols (-1, 1).
        """
        # Convert "010" -> [-1, 1, -1]
        return np.array([1 if b == '1' else -1 for b in bits_string])

    
    def to_passband(self, symbols: list, carrier_freq: float, 
                    sample_rate: float, samples_per_chip: int,
                    filter_type: str = "rrc", rolloff: float = 0.5) -> np.ndarray:
        """
        CDMA Passband Engine with optional Chip-Level RRC Filtering.


        NOTE: Do NOT normalize inside to_passband. Keep the math "pure" there. 
        Instead, handle the final volume scaling in your Transmitter class right 
        before sd.play() or sd.rec().
        """
        # Ensure we have a list of arrays (handles single-user edge case)
        if isinstance(symbols, np.ndarray):
            symbols = [symbols]

        num_bits = len(self.codes[0])
        total_chips = num_bits * self.sf
        
        # 1. Summation of all users in the "Chip Domain"
        # We use an impulse train approach for the chips to allow for RRC filtering
        num_samples = total_chips * samples_per_chip
        chip_impulses = np.zeros(num_samples)

        for uid in range(min(len(symbols), self.n_users)):
            user_bits = symbols[uid]  # These are BPSK symbols (-1, 1)
            
            # Use np.kron to turn each bit into a sequence of chips
            # e.g., if bit is 1 and code is [1, -1], it becomes [1, -1]
            user_chips = np.kron(user_bits, self.codes[uid])
            
            # Place these chips into the impulse train
            # We skip by samples_per_chip so each chip is an 'impulse'
            chip_impulses[::samples_per_chip] += user_chips

        # 2. THE FILTERING STEP (Optional RRC)
        if filter_type == "rect":
            # Square chips (Skye's original approach)
            rect_pulse = np.ones(samples_per_chip)
            baseband = signal.convolve(chip_impulses, rect_pulse, mode='same')
        else:
            # Rounded chips (Professional approach)
            rrc = self._get_rrc_coeffs(samples_per_chip, rolloff, span=10)
            baseband = signal.convolve(chip_impulses, rrc, mode='same')

        # 3. THE CARRIER STEP
        t = np.arange(len(baseband)) / sample_rate
        # Real-world audio: Baseband * Carrier
        passband = baseband * np.cos(2 * np.pi * carrier_freq * t)
                
        return passband.astype(np.float32)

    def _get_rrc_coeffs(self, sps: int, alpha: float, span: int = 10) -> np.ndarray:
            """Helper that wraps the partner's filter logic."""
            return rrc_filter(samples_per_symbol=sps, rolloff=alpha, span=span)

    def demodulate(self, soft_symbols: np.ndarray) -> str:
        """Hard decision for a single user's despread bits."""
        return "".join(['1' if s > 0 else '0' for s in soft_symbols])

    def despread(self, rx_chips: np.ndarray, user_id: int) -> np.ndarray:
        """
        Unique to CDMA: Extract a specific user's signal from the noise.
        """
        # Reshape to (number_of_bits, spreading_factor)
        n_bits = len(rx_chips) // self.sf
        reshaped = rx_chips[:n_bits*self.sf].reshape(n_bits, self.sf)
        
        # Multiply by the user's Walsh code and sum (Correlation)
        soft_bits = np.dot(reshaped, self.codes[user_id]) / self.sf
        return soft_bits

# ============================================================================
# FILTERS & UTILITIES
# ============================================================================
def rrc_filter(samples_per_symbol: int, rolloff: float = 0.5, span: int = 10) -> np.ndarray: 
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
    Base transmitter class handling common signal processing.
    """

    def __init__(self, modulation: Modulation, carrier_freq: float,
                 sample_rate: float, bit_rate: float, rolloff: float = 0.5):
        """
        Initialize transmitter.

        Args:
            modulation: Modulation instance (e.g., QPSKModulation())
            carrier_freq: Carrier frequency in Hz
            sample_rate: Sample rate in Hz
            baud_rate: Symbol rate in symbols/second
            rolloff: RRC rolloff factor

        Raises:
            ValueError: If sample_rate/baud_rate is not an integer
        """
        self.modulation = modulation
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.baud_rate = int(bit_rate / modulation.bits_per_symbol)
        self.samples_per_symbol = round(self.sample_rate / bit_rate)
        
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
        real_carrier = np.cos(2 * np.pi * self.carrier_freq * t)
        imag_carrier = np.sin(2 * np.pi * self.carrier_freq * t)
        
        modulated = baseband_signal.real * real_carrier - baseband_signal.imag * imag_carrier

        # Normalize to avoid clipping
        max_val = np.max(np.abs(modulated))
        if max_val > 0:
            modulated = modulated / max_val * 0.9

        return modulated.astype(np.float32)
    
    def transmit_bits(self, bits_input: Union[str, list]) -> np.ndarray:
        """

        Complete transmit chain: bits -> symbols -> baseband -> passband.
        (Refactored to handle both single-user (QPSK/FSK) and multi-user (CDMA))

        Args:
            bits: Bit string to transmit

        Returns:
            Real passband signal ready for audio transmission
        """
        if isinstance(bits_input, list):
            # CDMA Path: Convert list of bit-strings to list of symbol-arrays
            symbols = [self.modulation.modulate(b) for b in bits_input]
        else:
            # Standard Path: Convert single string to single symbol-array
            symbols = self.modulation.modulate(bits_input)
            
        # Use our refined 'to_passband' physics engine
        # This replaces upsample_and_filter and modulate_to_passband
        passband = self.modulation.to_passband(
            symbols, 
            self.carrier_freq, 
            self.sample_rate, 
            self.samples_per_symbol,
            filter_type="rrc",
            rolloff=self.rolloff
        )
        
        # FINAL NORMALIZATION: This protects your speakers and ensures consistency
        max_val = np.max(np.abs(passband))
        if max_val > 0:
            passband = (passband / max_val) * 0.8 # Scale to 80% volume
            
        return passband.astype(np.float32)

# ============================================================================
# RECEIVERS
# ============================================================================

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

class UnifiedReceiver(Receiver):
    def __init__(self, preamble_bits, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preamble_bits = preamble_bits

    def find_sync(self, passband_signal: np.ndarray):
        """
        Sliding window search to find the best timing offset (Samples) 
        and best symbol start.
        """
        best_sync = {"offset": 0, "errors": float('inf'), "symbols": None}
        
        ideal_preamble = self.modulation.modulate(self.preamble_bits)
        
        # 2. Search across one symbol period (Timing Sync)
        step = max(1, self.samples_per_symbol // 8)
        for offset in range(0, self.samples_per_symbol, step):
            baseband = self.demodulate_from_passband(passband_signal[offset:])
            all_symbols = self.filter_and_downsample(baseband)
            
            for s_idx in range(len(all_symbols) - len(ideal_preamble)):
                test_segment = all_symbols[s_idx : s_idx + len(ideal_preamble)]
                
                # Demodulate just this segment
                trial_bits = self.modulation.demodulate(test_segment)
                errors = sum(1 for b, p in zip(trial_bits, self.preamble_bits) if b != p)
                
                if errors < best_sync["errors"]:
                    best_sync = {
                        "offset": offset,
                        "start_symbol": s_idx,
                        "errors": errors,
                        "all_symbols": all_symbols
                    }
                    
            if best_sync["errors"] == 0: break # Found perfect lock
            
        return best_sync

    def receive_bits(self, passband_signal: np.ndarray) -> str:
        sync = self.find_sync(passband_signal)
        
        if sync["errors"] > len(self.preamble_bits) * 0.25:
            print("Warning: High preamble error. Sync may be unreliable.")
            
        # Extract payload (symbols after the preamble)
        payload_symbols = sync["all_symbols"][sync["start_symbol"] + len(self.modulation.modulate(self.preamble_bits)):]
        return self.modulation.demodulate(payload_symbols)

class CDMAReceiver(UnifiedReceiver):
    def receive_user_bits(self, passband_signal: np.ndarray, preamble_bits: str, user_id: int) -> str:
        # 1. Sync is done on the aggregate 'chip' stream
        sync = self.find_sync(passband_signal, preamble_bits)
        
        # 2. Extract the chips (In CDMA, 'symbols' from filter_and_downsample are actually 'chips')
        rx_chips = sync["all_symbols"]
        
        # 3. Use the CDMA-specific despreader
        soft_bits = self.modulation.despread(rx_chips, user_id)
        
        # 4. Hard decision
        return self.modulation.demodulate(soft_bits)

    def receive_bits(self, passband_signal: np.ndarray, sync_pattern: list) -> Tuple[str, int]:
        """
        Modified to include the 2D timing shift search.
        Returns (decoded_bits, best_offset_found)
        """
        best_bits = ""
        max_matches = -1
        best_offset = 0
        
        search_step = max(1, self.samples_per_symbol // 10) 
        
        for offset in range(0, self.samples_per_symbol, search_step):
            bb = self.demodulate_from_passband(passband_signal[offset:])
            trial_bits = self.modulation.demodulate(bb)
            
            # Correlation check: How many bits match the known sync pattern?
            matches = sum(1 for b, p in zip(trial_bits, cycle(sync_pattern)) if int(b) == p)
            
            if matches > max_matches:
                max_matches = matches
                best_bits = trial_bits
                best_offset = offset
                
        return best_bits, best_offset