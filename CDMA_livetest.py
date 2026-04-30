"""
Dual-Laptop CDMA Transceiver Test 
=================================================

Run this script on two laptops to conduct over-the-air testing:

LAPTOP A (Transmitter):
  $ python CDMA_livetest.py --mode tx

LAPTOP B (Receiver):
  $ python CDMA_livetest.py --mode rx

Each laptop displays its own local graphs independently.

TODO: convert this code into the Modular Class framework (this scheme is better than the one in modulation_framework_v2)

"""

import argparse
import numpy as np
import sounddevice as sd

FS = 51200
FC = 12800
BIT_RATE = 100
SF = 64

CHIP_RATE = BIT_RATE * SF
SPC = FS // CHIP_RATE

AMPLITUDE = 0.35
INPUT_DEVICE = 1
RX_SECONDS = 8.0

LEAD_GAP_SEC = 0.5
TAIL_GAP_SEC = 0.5

PREAMBLE_BITS = np.tile([1, 0, 1, 1, 0, 0, 1, 0], 16)   # 128 bits
PAYLOAD_BITS = np.tile([0, 0, 0, 1, 1, 1, 1, 0], 16) 

FS = 48_000
BIT_RATE = 500
SPB = int(FS / BIT_RATE)

F0 = 1000
F1 = 5000

# ============================================================================
# TX
# ============================================================================

def walsh(n):
    H = np.array([[1]], dtype=int)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


CODES = walsh(SF)
USER_CODE = CODES[7]


def bpsk(bits):
    return 1 - 2 * np.asarray(bits, dtype=float)


def spread(bits):
    return np.outer(bpsk(bits), USER_CODE).reshape(-1)


PREAMBLE_CHIPS = spread(PREAMBLE_BITS)
PAYLOAD_CHIPS = spread(PAYLOAD_BITS)


def make_audio():
    chips = np.concatenate([PREAMBLE_CHIPS, PAYLOAD_CHIPS])
    AMPLITUDEles = np.repeat(chips, SPC)

    n = np.arange(len(AMPLITUDEles))
    audio = AMPLITUDEles * np.cos(2 * np.pi * FC * n / FS)

    fade = int(0.01 * FS)
    if len(audio) > 2 * fade:
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)

    audio = AMPLITUDE * audio / (np.max(np.abs(audio)) + 1e-12)

    lead = np.zeros(int(LEAD_GAP_SEC * FS), dtype=np.float32)
    tail = np.zeros(int(TAIL_GAP_SEC * FS), dtype=np.float32)

    return np.concatenate([lead, audio.astype(np.float32), tail])

def tx():
    audio = make_audio()

    print("CDMA TX")
    print("64-chip temporal CDMA")
    print(f"Carrier: {FC} Hz")
    print(f"Bit rate: {BIT_RATE} bps")
    print(f"Chip rate: {CHIP_RATE} chips/sec")
    print(f"SPC: {SPC}")
    print(f"Frame length: {len(audio) / FS:.2f} sec")

    sd.play(audio, FS)
    sd.wait()
    print("sent once")
    
# ============================================================================
# RX
# ============================================================================

def record_audio():
    print(f"Recording {RX_SECONDS:.2f} sec from input device {INPUT_DEVICE}...")

    r = sd.rec(
        int(RX_SECONDS * FS),
        AMPLITUDElerate=FS,
        channels=2,
        dtype="float32",
        device=INPUT_DEVICE,
    )
    sd.wait()

    ch0 = r[:, 0]
    ch1 = r[:, 1]

    print("RX max ch0:", np.max(np.abs(ch0)))
    print("RX max ch1:", np.max(np.abs(ch1)))

    return ch0 if np.max(np.abs(ch0)) >= np.max(np.abs(ch1)) else ch1


def downconvert(audio, offset):
    audio = np.asarray(audio, dtype=float)
    audio = audio - np.mean(audio)
    audio = audio[offset:]

    n = np.arange(len(audio))
    bb = 2 * audio * np.exp(-1j * 2 * np.pi * FC * n / FS)

    usable = (len(bb) // SPC) * SPC
    bb = bb[:usable]

    return bb.reshape(-1, SPC).mean(axis=1)


def despread(chips):
    n_bits = len(chips) // SF
    chips = chips[:n_bits * SF].reshape(n_bits, SF)

    soft = np.real(chips @ USER_CODE)
    bits = (soft < 0).astype(int)

    return bits, soft


def phase_correct(chips, known_chips):
    alpha = np.vdot(known_chips, chips) / (np.vdot(known_chips, known_chips) + 1e-12)
    return chips / (alpha + 1e-12), alpha


def decode(audio):
    best = None

    pre_len = len(PREAMBLE_CHIPS)
    payload_len = len(PAYLOAD_CHIPS)

    for offset in range(SPC):
        chips = downconvert(audio, offset)

        if len(chips) < pre_len + payload_len:
            continue

        corr = np.correlate(chips, PREAMBLE_CHIPS, mode="valid")
        metrics = np.abs(corr) / (pre_len * (np.std(chips) + 1e-12))

        # Try lots of possible locks. This avoids fake-lock death.
        top = np.argsort(metrics)[-1000:]

        for k in top:
            k = int(k)
            payload_start = k + pre_len
            payload_end = payload_start + payload_len

            if payload_end > len(chips):
                continue

            pre_rx = chips[k:k + pre_len]
            payload_rx = chips[payload_start:payload_end]

            pre_corr, alpha = phase_correct(pre_rx, PREAMBLE_CHIPS)
            payload_corr = payload_rx / (alpha + 1e-12)

            pre_bits, _ = despread(pre_corr)
            payload_bits, _ = despread(payload_corr)

            pre_errors = int(np.sum(pre_bits[:len(PREAMBLE_BITS)] != PREAMBLE_BITS))
            payload_errors = int(np.sum(payload_bits[:len(PAYLOAD_BITS)] != PAYLOAD_BITS))

            score = (payload_errors + 2 * pre_errors, payload_errors, -float(metrics[k]))

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "offset": offset,
                    "k": k,
                    "metric": float(metrics[k]),
                    "alpha": alpha,
                    "pre_errors": pre_errors,
                    "payload_errors": payload_errors,
                    "bits": payload_bits,
                    "n": len(PAYLOAD_BITS),
                    "pe": payload_errors / len(PAYLOAD_BITS),
                }

    return best

def rx():
    audio = record_audio()
    result = decode(audio)

    if result is None:
        print("No frame found.")
        return

    print("CDMA RX")
    print(f"best AMPLITUDEle offset: {result['offset']}")
    print(f"preamble chip index: {result['k']}")
    print(f"sync metric: {result['metric']:.3f}")
    print(f"channel magnitude: {abs(result['alpha']):.6f}")
    print(f"channel phase: {np.angle(result['alpha']):.3f} rad")
    print(f"preamble errors: {result['pre_errors']}/{len(PREAMBLE_BITS)}")
    print(f"errors: {result['payload_errors']}/{result['n']}")
    print(f"Pe: {result['pe']:.5f}")
    print("first 64 rx bits:")
    print(result["bits"][:64])


def loopback():
    result = decode(make_audio())

    print("LOOPBACK")
    print(f"preamble errors: {result['pre_errors']}/{len(PREAMBLE_BITS)}")
    print(f"errors: {result['payload_errors']}/{result['n']}")
    print(f"Pe: {result['pe']:.5f}")

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