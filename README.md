# ENGN 1580: Communications Systems Final Project

by Milan Capoor, Emmett Forrestel, Tomas Rivera, Thomas Seidel, Skye Stewart

[Final Presentation](https://docs.google.com/presentation/d/1AdmPKkHYuwo9AYuu_BDBRQR0ejneYA-vkrMkIiHwWOw/edit?usp=sharing)

## General Remarks

This repository contains two main groups of files: those already integrated into the Modular Class framework and those more-accurate standalone `*_livetest.py` scripts for the live class demonstration.

Our goal was to create a completely modular and extensible system where the modulation was separated from the transmitter and receiver logic which is separated from the I/O and the visualization. Due to time constraints, only the simulated channel models are fully integrated. Currently, `modulation_framework_v2.py` and `modulation_framework.py` contain modulation schemes that are reimplemented in the livetest scripts.

We apologize for the tech debt, the spaghetti code, and the varying levels of functionality.

## Directory Structure and Usage

1. Live test files (for two laptops to test via audio), used in class demonstration:
    - `BPSK_livetest.py`
    - `QPSK_livetest.py`
    - `CDMA_livetest.py`
    - `listener.py` (a simple script to visualize the incoming audio signal in the time and frequency domains. Supports live plotting with pause and zoom. Eventually will include a constellation map that will display the received symbols in real time to visually identify the modulation scheme used)

    These files should be run via the command line on two separate laptops with a microphone and speaker. To transmit, run `$ python *_livetest.py --mode tx` and to receive, run `$ python *_livetest.py --mode rx`.

2. Modular Class libraries:
    - `modulation_framework.py` (the main library for modulation schemes and transmitter/receiver classes)
    - `modulation_framework_v2.py` (a refactored version of the main library with better timing sync but not fully integrated/debugged yet. The modulation classes in this file are interchangeable supersede those in `modulation_framework.py` but the transmitter and receiver classes are not yet fully integrated into the live test scripts. However, the live test modulation scripts are even better than the ones here in v2 so should be integrated soon)
    - `simulation_framework.py` (library used to simulate the channel, provide a unified transceiver dashboard, and wrap the modulation framework classes with buffered I/O for real time visualization)

3. Modular Class test files (used for the simulation results in the presentation):
    - `closed_loop_test.py` (unified script to package the modular modulation schemes and multi-purpose transceiver classes with channel simulation and live visualization.)
    - `dual_laptop_test.py` (a still in progress wrapper for the modular classes that pipes their I/O to device audio and provides real-time plotting. Run in the command line with `$ python dual_laptop_test.py --mode tx` to transmit,  `$ python dual_laptop_test.py --mode rx` to receive, or `$ python dual_laptop_test.py --mode loopback` to launch both nodes. This script is still in progress and facing some major timing and threading issues, especially on the receiver. Computaton is especially inefficient and running loopback mode may cause freezing on some machines)
