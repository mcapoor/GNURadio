'''
Main wrapper file for pick-your-own-adventure modular testing over simulated channel
=============================

- To change the modulation, select one of the commented options in lines 56-59 (or add your own)
- To change the channel conditions, change the parameters in lines 46-49
- To run the BER sweep test, set run_ber_sweep = True in line 51 (this runs only after you close the real-time plots)

TODO: The Transmitter and UnifiedReceiver classes in modulation_framework_v2 implement better timing sync but aren't fully integrated/debugged yet. The classes in modulation_framework are out of date but currently work (in simulation only). Eventually, we should deprecate modulation_framework and switch everything over to modulation_framework_v2, but for now you can mix and match as needed. The modulation classes provided by modulation_framework_v2 are all interchangeable between the two frameworks since they have the same interface.

'''

from modulation_framework_v2 import ( 
    Transmitter, UnifiedReceiver,
    nQAMModulation, 
    PSKModulation, 
    GrayCodedQPSK, 
    CDMAModulation, CDMAReceiver
) 

import modulation_framework as milan

from simulation_framework import (
    channel_model,
    bit_stream_generator,
    StreamingTransceiver,
    create_animated_transceiver_plot,
    ber_vs_snr_sweep
)

import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

plt.ion()


# ============================================================================
# CONFIGURATION 
# ============================================================================

CARRIER_FREQ = 12_800  # Hz
SAMPLE_RATE = 44_100
BITS_PER_SECOND = 5_000

SNR = 10 # dB
CHANNEL_DELAY = 0
CHANNEL_FREQ_OFFSET = 0
CHANNEL_PHASE_JITTER = 0

rng = np.random.default_rng(12345)
PREAMBLE_BITS = rng.integers(0, 2, size=128, dtype=int)
BIT_STREAM_BIAS = 0.5 # Probability of bit=1 in the random bit stream

run_ber_sweep = False

# ============================================================================
# MAIN LOOP 
# ============================================================================

# modulation = nQAMModulation(16)  
modulation = PSKModulation(f0_offset=-10000, f1_offset=10000, fs=SAMPLE_RATE, spb=int(SAMPLE_RATE/BITS_PER_SECOND)) 
# modulation = GrayCodedQPSK()
# modulation = CDMAModulation(n_users=4, spreading_factor=64)

transmitter = milan.Transmitter(
    modulation=modulation,
    carrier_freq=CARRIER_FREQ,
    sample_rate=SAMPLE_RATE,
    bit_rate=BITS_PER_SECOND
)

receiver = milan.Receiver(
    modulation=modulation,
    carrier_freq=CARRIER_FREQ,
    sample_rate=SAMPLE_RATE,
    bit_rate=BITS_PER_SECOND
)

def channel(signal): 
    return channel_model(
        signal,
        snr_db=SNR,
        sample_rate=SAMPLE_RATE,
        delay_samples=CHANNEL_DELAY,
        freq_offset=CHANNEL_FREQ_OFFSET,
        phase_jitter=CHANNEL_PHASE_JITTER
    ) 
    
streamer = StreamingTransceiver(
    transmitter=transmitter,
    receiver=receiver,
    channel_model_fn=channel,
    sample_rate=SAMPLE_RATE,
    snr_db=SNR,
    bits_per_symbol=modulation.bits_per_symbol, 
)

source = bit_stream_generator(BIT_STREAM_BIAS, chunk_size=100)

# ============================================================================
# TESTING
# ============================================================================

print("\n" + "="*60)
print("Starting Real-Time Streaming Mode")
print("="*60)
print(f"Streaming {BITS_PER_SECOND} bits/second")
print(f"Channel SNR: {SNR} dB")
print(f"Carrier: {CARRIER_FREQ} Hz, Sample Rate: {SAMPLE_RATE} Hz")
print("Close plots to continue to BER sweep test...")
print("="*60 + "\n")

fig, anim = create_animated_transceiver_plot(
    streamer,
    source,
    update_interval_ms=500, 
    max_frames=None 
)

plt.show(block=True) 

if run_ber_sweep:
    print("="*60)
    print("BER vs SNR Sweep (50, 500, 5000 bps)")
    print("="*60)

    ber_vs_snr_sweep(
        transmitter, receiver, channel_model,
        bit_rates=[50, 500, 5000],
        snr_range=np.arange(-21, 21, 3),
        num_bits=1000,
        sample_rate=SAMPLE_RATE, 
        bits_per_symbol=modulation.bits_per_symbol
    )

    plt.show(block=True)
