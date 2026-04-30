"""
QPSK Transceiver Test 
=================================================

Run this script on two laptops to conduct over-the-air testing:

LAPTOP A (Transmitter):
  $ python QPSK_livetest.py --mode tx

LAPTOP B (Receiver):
  $ python QPSK_livetest.py --mode rx

Each laptop displays its own local graphs independently.

TODO: convert this code into the Modular Class framework (this scheme is better than the one in modulation_framework_v2)
"""

# ============================================================================
# CONFIGURATION
# ============================================================================
from queue import Empty, Queue
import time
import argparse

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

FS = 48_000
BIT_RATE = 4800
CARRIER = 4000

INPUT_DEVICE = 7 # depends on your system; use sd.query_devices() to find the right one for your microphone

BITS_PER_SYMBOL = 2
SYMBOL_RATE = BIT_RATE / BITS_PER_SYMBOL
SPS = int(FS / SYMBOL_RATE)

CHUNK_SEC = 3.0
N = int(FS * CHUNK_SEC)

LPF_WINDOW = 5
MIN_RX_AMP = 0.003
MAX_PREAMBLE_ERRORS = 12

AMPLITUDE = 0.35

PAYLOAD_SEC = 0.5
PAYLOAD_SYMBOLS = int(SYMBOL_RATE * PAYLOAD_SEC)

DISCARD_PAYLOAD_START = 2
DISCARD_PAYLOAD_END = 4

LOOP_GAIN = 0.03

rng = np.random.default_rng(12345)
PREAMBLE_BITS = rng.integers(0, 2, size=128, dtype=int)
PREAMBLE_SYMBOLS = len(PREAMBLE_BITS) // 2

PAYLOAD_PATTERN = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)

PLOT_UPDATE_INTERVAL_MS = 500
LIVE_PLOT_UPDATE_MS = 50

# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def bits_to_iq(b0, b1):
    if b0 == 0 and b1 == 0:
        return 1, 1
    if b0 == 0 and b1 == 1:
        return -1, 1
    if b0 == 1 and b1 == 1:
        return -1, -1
    return 1, -1


def bits_to_symbols(bits):
    symbols = []

    for k in range(0, len(bits), 2):
        I, Q = bits_to_iq(bits[k], bits[k + 1])
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

def moving_average(x, window):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ============================================================================
# RX
# ============================================================================

def decode_symbols(z):
    bits = []

    for val in z:
        bits.extend(iq_to_bits(np.real(val), np.imag(val)))

    return np.array(bits, dtype=int)


def best_payload_alignment(bits_hat):
    best_errors = None
    best_expected = None
    best_shift = None
    best_inverted = None

    for shift_bits in range(len(PAYLOAD_PATTERN)):
        expected = np.resize(np.roll(PAYLOAD_PATTERN, -shift_bits), len(bits_hat))

        for inverted in [False, True]:
            candidate = 1 - expected if inverted else expected
            errors = np.sum(bits_hat != candidate)

            if best_errors is None or errors < best_errors:
                best_errors = errors
                best_expected = candidate
                best_shift = shift_bits
                best_inverted = inverted

    return best_shift, best_inverted, best_errors, best_expected


def find_preamble(z):
    known = bits_to_symbols(PREAMBLE_BITS)
    n_known = np.arange(PREAMBLE_SYMBOLS)

    best = None
    max_start = len(z) - PREAMBLE_SYMBOLS - PAYLOAD_SYMBOLS

    if max_start <= 0:
        return None

    for start in range(max_start):
        test = z[start:start + PREAMBLE_SYMBOLS]

        phase_err = np.angle(test * np.conj(known))
        phase_err = np.unwrap(phase_err)

        phase_slope, phase0 = np.polyfit(n_known, phase_err, 1)

        correction = np.exp(-1j * (phase0 + phase_slope * n_known))
        test_corr = test * correction

        bits_hat = decode_symbols(test_corr)
        errors = np.sum(bits_hat != PREAMBLE_BITS)

        if best is None or errors < best["errors"]:
            best = {
                "start": start,
                "phase0": phase0,
                "phase_slope": phase_slope,
                "errors": errors,
            }

    return best


def decision_directed_phase_track(z_linear, loop_gain=LOOP_GAIN):
    z_corr = np.zeros_like(z_linear, dtype=complex)
    loop_phase = 0.0

    for k in range(len(z_linear)):
        z_tmp = z_linear[k] * np.exp(-1j * loop_phase)
        z_corr[k] = z_tmp

        I = 1 if np.real(z_tmp) >= 0 else -1
        Q = 1 if np.imag(z_tmp) >= 0 else -1
        ref = (I + 1j * Q) / np.sqrt(2)

        phase_error = np.angle(z_tmp * np.conj(ref))
        loop_phase += loop_gain * phase_error

    return z_corr


def receive_chunk():
    print("Recording...")

    audio_queue = Queue()
    recorded_blocks = []

    live_fig, live_ax = plt.subplots(figsize=(10, 3))
    (live_line,) = live_ax.plot([], [], linewidth=0.8, color="tab:blue")
    live_ax.set_title("Live recording")
    live_ax.set_xlabel("Time [s]")
    live_ax.set_ylabel("Amplitude")
    live_ax.set_xlim(0, CHUNK_SEC)
    live_ax.set_ylim(-1.0, 1.0)
    live_ax.grid(True, alpha=0.3)
    live_status = live_ax.text(
        0.01,
        0.95,
        "Recording...",
        transform=live_ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def callback(indata, frames, time_info, status):
        if status:
            print(status)

        audio_queue.put(indata[:, 0].copy())

    start_time = time.monotonic()

    try:
        with sd.InputStream(
            samplerate=FS,
            channels=1,
            dtype="float32",
            device=INPUT_DEVICE,
            callback=callback,
            blocksize=1024,
        ):
            while time.monotonic() - start_time < CHUNK_SEC:
                while True:
                    try:
                        recorded_blocks.append(audio_queue.get_nowait())
                    except Empty:
                        break

                if recorded_blocks:
                    r_live = np.concatenate(recorded_blocks)
                    t_live = np.arange(len(r_live)) / FS
                    live_line.set_data(t_live, r_live)
                    live_ax.set_xlim(0, max(CHUNK_SEC, t_live[-1] if len(t_live) else CHUNK_SEC))

                    live_peak = max(np.max(np.abs(r_live)), MIN_RX_AMP)
                    live_ax.set_ylim(-1.2 * live_peak, 1.2 * live_peak)
                    live_status.set_text(
                        f"Recording... {min(time.monotonic() - start_time, CHUNK_SEC):.1f}/{CHUNK_SEC:.1f} s"
                    )

                    live_fig.canvas.draw_idle()

                plt.pause(LIVE_PLOT_UPDATE_MS / 1000.0)
    finally:
        while True:
            try:
                recorded_blocks.append(audio_queue.get_nowait())
            except Empty:
                break

        plt.close(live_fig)

    if recorded_blocks:
        r = np.concatenate(recorded_blocks)
    else:
        r = np.zeros(N, dtype=np.float32)

    if len(r) < N:
        r = np.pad(r, (0, N - len(r)))

    r = r[:N]

    rx_amp = np.max(np.abs(r))
    print("RX max amplitude:", rx_amp)

    if rx_amp < MIN_RX_AMP:
        print("Signal too weak.")
        return None

    n = np.arange(len(r))
    t = n / FS

    I_raw = 2 * r * np.cos(2 * np.pi * CARRIER * t)
    Q_raw = -2 * r * np.sin(2 * np.pi * CARRIER * t)

    I_filt = moving_average(I_raw, LPF_WINDOW)
    Q_filt = moving_average(Q_raw, LPF_WINDOW)

    best = None

    for offset in range(0, SPS, 1):
        idx = np.arange(offset, len(r), SPS)
        z = I_filt[idx] + 1j * Q_filt[idx]

        if len(z) < PREAMBLE_SYMBOLS + PAYLOAD_SYMBOLS:
            continue

        cand = find_preamble(z)

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
    print("Preamble errors:", best["errors"], "/", len(PREAMBLE_BITS))

    if best["errors"] > MAX_PREAMBLE_ERRORS:
        print("Preamble lock not clean enough; skipping chunk.")
        return None

    n_sym = np.arange(len(best["z"])) - best["start"]
    phase_track = best["phase0"] + best["phase_slope"] * n_sym

    z_linear = best["z"] * np.exp(-1j * phase_track)
    z_corr = decision_directed_phase_track(z_linear)

    payload_start = best["start"] + PREAMBLE_SYMBOLS
    payload_z = z_corr[payload_start:payload_start + PAYLOAD_SYMBOLS]
    payload_z = payload_z[DISCARD_PAYLOAD_START:-DISCARD_PAYLOAD_END]

    if len(payload_z) == 0:
        print("No payload symbols.")
        return None

    bits_hat = decode_symbols(payload_z)
    shift, inverted, errors, bits_expected = best_payload_alignment(bits_hat)

    print("Best payload bit shift:", shift)
    print("Payload inverted:", inverted)

    return {
        "r": r,
        "z_payload": payload_z,
        "bits_hat": bits_hat,
        "bits_expected": bits_expected,
        "errors": errors,
        "audio": r,
        "num_bits": len(bits_hat),
    }


def rx():
    print("QPSK RX ThinkPad with decision-directed phase tracking")
    print(f"Input device = {INPUT_DEVICE}")
    print(f"FS = {FS}")
    print(f"Bit rate = {BIT_RATE}")
    print(f"Symbol rate = {SYMBOL_RATE}")
    print(f"SPS = {SPS}")
    print(f"Carrier = {CARRIER}")
    print(f"Chunk sec = {CHUNK_SEC}")
    print(f"LPF window = {LPF_WINDOW}")
    print(f"Preamble bits = {len(PREAMBLE_BITS)}")
    print(f"Payload symbols = {PAYLOAD_SYMBOLS}")
    print(f"Loop gain = {LOOP_GAIN}")
    print("First 32 preamble bits:")
    print(PREAMBLE_BITS[:32])

    total_errors = 0
    total_bits = 0

    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    
    try:
        while True:
            result = receive_chunk()

            if result is None:
                continue

            total_errors += result["errors"]
            total_bits += result["num_bits"]

            Pe_chunk = result["errors"] / result["num_bits"]
            Pe_running = total_errors / total_bits

            print(f"Chunk Pe:   {Pe_chunk:.5f}")
            print(f"Running Pe: {Pe_running:.5f}")
            print("First 40 decoded bits:")
            print(result["bits_hat"][:40])
            print("First 40 expected bits:")
            print(result["bits_expected"][:40])

            update_rx_plots(fig, axes, result, Pe_running)

    except KeyboardInterrupt:
        print("\nStopped.")

        if total_bits > 0:
            print(f"Final Pe: {total_errors / total_bits:.6f}")
            print(f"Total errors: {total_errors}")
            print(f"Total bits: {total_bits}")


                    
                    
def update_rx_plots(fig, axes, result, Pe_running):
    r = result["r"]
    z = result["z_payload"]

    bits_hat = result["bits_hat"]
    bits_expected = result["bits_expected"]
    audio = result["audio"]
    symbols = result["z_payload"]

    ax_wave, ax_spec, ax_const, ax_bits = axes

    t = np.arange(len(r)) / FS

    ax_wave.clear()
    ax_wave.plot(t, r)
    ax_wave.set_title("RX waveform")
    ax_wave.set_xlabel("Time [s]")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True)

    R = np.fft.rfft(r * np.hanning(len(r)))
    freqs = np.fft.rfftfreq(len(r), 1 / FS)
    mag = 20 * np.log10(np.abs(R) + 1e-12)
    mask = (freqs >= 20) & (freqs <= 20_000)

    ax_spec.clear()
    ax_spec.plot(freqs[mask], mag[mask])
    ax_spec.axvline(CARRIER, label=f"Carrier = {CARRIER} Hz")
    ax_spec.set_title("RX spectrum, 20 Hz–20 kHz")
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel("Magnitude [dB]")
    ax_spec.legend()
    ax_spec.grid(True)

    ax_const.clear()
    ax_const.scatter(np.real(z), np.imag(z), s=10)
    ax_const.axhline(0)
    ax_const.axvline(0)
    ax_const.set_aspect("equal", adjustable="box")
    ax_const.set_title(f"QPSK constellation | Running Pe = {Pe_running:.5f}")
    ax_const.set_xlabel("I")
    ax_const.set_ylabel("Q")
    ax_const.grid(True)

    m = max(
        np.max(np.abs(np.real(z))),
        np.max(np.abs(np.imag(z))),
        1e-6,
    )
    ax_const.set_xlim(-1.2 * m, 1.2 * m)
    ax_const.set_ylim(-1.2 * m, 1.2 * m)

    n_show = min(120, len(bits_hat))

    ax_bits.clear()
    ax_bits.step(np.arange(n_show), bits_hat[:n_show], where="mid", label="decoded")
    ax_bits.step(np.arange(n_show), bits_expected[:n_show] + 0.05, where="mid", label="expected")
    ax_bits.set_ylim(-0.2, 1.3)
    ax_bits.set_title("Decoded vs expected bits")
    ax_bits.set_xlabel("Bit index")
    ax_bits.legend()
    ax_bits.grid(True)

    fig.suptitle(f"RX analysis | Running Pe = {Pe_running:.5f}")
    fig.canvas.draw_idle()
    plt.pause(0.001)

# ============================================================================
# TX
# ===========================================================================




def make_frame():
    payload_bits = np.resize(PAYLOAD_PATTERN, 2 * PAYLOAD_SYMBOLS)
    bits = np.concatenate([PREAMBLE_BITS, payload_bits])

    symbols = bits_to_symbols(bits)

    I = np.repeat(np.real(symbols), SPS)
    Q = np.repeat(np.imag(symbols), SPS)

    n = np.arange(len(I))
    t = n / FS

    audio = I * np.cos(2 * np.pi * CARRIER * t) - Q * np.sin(2 * np.pi * CARRIER * t)

    audio = audio / np.max(np.abs(audio))
    audio = AMPLITUDE * audio

    return audio.astype(np.float32)


def tx():
    print("QPSK TX MAC")
    print(f"FS = {FS}")
    print(f"Bit rate = {BIT_RATE}")
    print(f"Symbol rate = {SYMBOL_RATE}")
    print(f"SPS = {SPS}")
    print(f"Carrier = {CARRIER}")
    print(f"Amplitude = {AMPLITUDE}")
    print(f"Preamble bits = {len(PREAMBLE_BITS)}")
    print(f"Payload symbols = {PAYLOAD_SYMBOLS}")
    print("First 32 preamble bits:")
    print(PREAMBLE_BITS[:32])

    frame = make_frame()
    silence = np.zeros(int(0.15 * FS), dtype=np.float32)
    tx = np.concatenate([frame, silence])

    try:
        while True:
            sd.play(tx, FS)
            sd.wait()

    except KeyboardInterrupt:
        print("\nStopped.")

# ============================================================================
# MAIN LOOP
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tx", "rx"], required=True)
    args = parser.parse_args()

    if args.mode == "tx":
        tx()
    elif args.mode == "rx":
        rx()
    else:
        raise ValueError("Invalid mode try --mode tx|rx")