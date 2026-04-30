"""
Dual-Laptop binary phase shift keying Transceiver Test 
=================================================

Run this script on two laptops to conduct over-the-air testing:

LAPTOP A (Transmitter):
  $ python BPSK_livetest.py --mode tx

LAPTOP B (Receiver):
  $ python BPSK_livetest.py --mode rx

Each laptop displays its own local graphs independently.

TODO: convert this code into the Modular Class framework (this scheme is better than the one in modulation_framework_v2)

"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import argparse

FS = 48_000
BIT_RATE = 500
SPB = int(FS / BIT_RATE)
AMPLITUDE = 0.35

F0 = 1000
F1 = 5000
INPUT_DEVICE = 6

CHUNK_SEC = 3.0
N = int(FS * CHUNK_SEC)

MIN_RX_AMP = 0.0005
MAX_PREAMBLE_ERRORS = 12

rng = np.random.default_rng(12345)
PREAMBLE_BITS = rng.integers(0, 2, size=128, dtype=int)

PAYLOAD_BITS = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=int)

# ============================================================================
# RX 
# ============================================================================

def tone_energy(x, f):
    x = x - np.mean(x)

    n = np.arange(len(x))
    c = np.cos(2 * np.pi * f * n / FS)
    s = np.sin(2 * np.pi * f * n / FS)

    I = np.sum(x * c)
    Q = np.sum(x * s)

    return I * I + Q * Q


def decode_at_offset(r, offset):
    bits = []
    scores = []

    i = offset
    while i + SPB <= len(r):
        chunk = r[i:i + SPB]

        e0 = tone_energy(chunk, F0)
        e1 = tone_energy(chunk, F1)

        score = np.log((e1 + 1e-12) / (e0 + 1e-12))
        bit = 1 if score > 0 else 0

        bits.append(bit)
        scores.append(score)

        i += SPB

    return np.array(bits, dtype=int), np.array(scores)


def find_preamble(bits):
    best = None

    for start in range(0, len(bits) - len(PREAMBLE_BITS)):
        test = bits[start:start + len(PREAMBLE_BITS)]
        errors = np.sum(test != PREAMBLE_BITS)

        if best is None or errors < best["errors"]:
            best = {
                "start": start,
                "errors": errors,
            }

    return best


def best_payload_alignment(bits_hat):
    best_errors = None
    best_expected = None
    best_shift = None

    for shift in range(len(PAYLOAD_BITS)):
        expected = np.resize(np.roll(PAYLOAD_BITS, -shift), len(bits_hat))
        errors = np.sum(bits_hat != expected)

        if best_errors is None or errors < best_errors:
            best_errors = errors
            best_expected = expected
            best_shift = shift

    return best_shift, best_errors, best_expected


def receive():
    print("Recording...")

    r = sd.rec(
        N,
        samplerate=FS,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE,
    )
    sd.wait()
    r = r[:, 0]

    amp = np.max(np.abs(r))
    print("RX max amplitude:", amp)

    if amp < MIN_RX_AMP:
        print("Signal too weak.")
        return None

    best = None

    for offset in range(0, SPB, max(1, SPB // 100)):
        bits, scores = decode_at_offset(r, offset)

        if len(bits) < len(PREAMBLE_BITS) + 20:
            continue

        pre = find_preamble(bits)

        candidate = {
            "offset": offset,
            "preamble_start": pre["start"],
            "preamble_errors": pre["errors"],
            "bits_all": bits,
            "scores_all": scores,
        }

        if best is None or candidate["preamble_errors"] < best["preamble_errors"]:
            best = candidate

    if best is None:
        print("Could not find preamble.")
        return None

    print("Best offset:", best["offset"])
    print("Preamble start bit:", best["preamble_start"])
    print("Preamble errors:", best["preamble_errors"], "/", len(PREAMBLE_BITS))

    if best["preamble_errors"] > MAX_PREAMBLE_ERRORS:
        print("Preamble lock not clean enough.")
        return None

    payload_start = best["preamble_start"] + len(PREAMBLE_BITS)
    payload_bits = best["bits_all"][payload_start:]
    payload_scores = best["scores_all"][payload_start:]

    payload_bits = payload_bits[:-10]
    payload_scores = payload_scores[:-10]

    if len(payload_bits) == 0:
        print("No payload bits.")
        return None

    shift, errors, expected = best_payload_alignment(payload_bits)

    return {
        "bits": payload_bits,
        "scores": payload_scores,
        "expected": expected,
        "errors": errors,
        "num_bits": len(payload_bits),
        "shift": shift,
    }


def rx():
    print("BFSK RX")
    print(f"Bit rate = {BIT_RATE}")
    print(f"SPB = {SPB}")
    print(f"F0 = {F0}")
    print(f"F1 = {F1}")
    print(f"Input device = {INPUT_DEVICE}")
    print("First 32 preamble bits:")
    print(PREAMBLE_BITS[:32])

    total_errors = 0
    total_bits = 0

    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    try:
        while True:
            result = receive()

            if result is None:
                continue

            total_errors += result["errors"]
            total_bits += result["num_bits"]

            pe_chunk = result["errors"] / result["num_bits"]
            pe_running = total_errors / total_bits

            print("Best payload shift:", result["shift"])
            print(f"Chunk Pe:   {pe_chunk:.5f}")
            print(f"Running Pe: {pe_running:.5f}")
            print("First 40 decoded:")
            print(result["bits"][:40])
            print("First 40 expected:")
            print(result["expected"][:40])

            axes[0].clear()
            axes[0].plot(result["scores"])
            axes[0].axhline(0)
            axes[0].set_title("FSK score: log(E1/E0)")
            axes[0].grid(True)

            n_show = min(120, len(result["bits"]))

            axes[1].clear()
            axes[1].step(np.arange(n_show), result["bits"][:n_show], where="mid", label="decoded")
            axes[1].step(np.arange(n_show), result["expected"][:n_show] + 0.05, where="mid", label="expected")
            axes[1].set_ylim(-0.2, 1.3)
            axes[1].set_title(f"BFSK bits | Running Pe = {pe_running:.5f}")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            plt.pause(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")
        if total_bits > 0:
            print(f"Final Pe: {total_errors / total_bits:.6f}")
            print(f"Total errors: {total_errors}")
            print(f"Total bits: {total_bits}")

# ============================================================================
# TX
# ============================================================================

def make_audio():
    payload = np.resize(PAYLOAD_BITS, PAYLOAD_BITS)
    bits = np.concatenate([PREAMBLE_BITS, payload])

    n = np.arange(SPB)
    tone0 = np.sin(2 * np.pi * F0 * n / FS)
    tone1 = np.sin(2 * np.pi * F1 * n / FS)

    audio = np.concatenate([tone1 if b else tone0 for b in bits])
    audio = audio / np.max(np.abs(audio))
    audio = AMPLITUDE * audio

    return audio.astype(np.float32)


def tx():
    print("BFSK TX")
    print(f"Bit rate = {BIT_RATE}")
    print(f"SPB = {SPB}")
    print(f"F0 = {F0}")
    print(f"F1 = {F1}")
    print("First 32 preamble bits:")
    print(PREAMBLE_BITS[:32])

    frame = make_audio()
    gap = np.zeros(int(0.25 * FS), dtype=np.float32)
    tx = np.concatenate([frame, gap])

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