# ENGN 1580: Communications Systems Final Project

## Installation and setup

Install [gnuradio](https://wiki.gnuradio.org/index.php?title=InstallingGR), clone the repo, and open ``main.grc``

## Currently implemented features 
- Modulating a pre-specified bit string by a carrier cosine wave
- Outputting that wave to audio (default ``pipewire`` to run to on Arch), and time-domain and frequency domain live graphs

## Current bugs
- The current visualization pattern is laggy and not configured to the right time interval. Ideally, we should set the message and modulated waveform time dispaly graphs to show the entire input bit stream at once

## Future plans
- Binary signaling (with whatever symbol waveform you'd like)
- Binary signaling using a raised cosine pulse as the symbol waveform
- Receiver and minimum error decision rule
- Other modulation schemes
- Synchronization between transmitter and receiver devices
- Image transmission
- Equalization and echo correction
- Error correction encodings
- Protection against noise and adversarial interference 
