"""
Out-of-the-loop real-time visualization for transmitter signals. 
================================

Displays time-domain waveform, frequency spectrum, and constellation of the transmitted signal in real-time.

TODO: the constellation plot is supposed to follow our standard receive path and display whatever constellation is received. Right now, it just plots lots of symbols around the origin 
"""


import sounddevice as sd 
import numpy as np
import matplotlib.pyplot as plt
import queue
from collections import deque
from scipy.signal import convolve

CHUNK_SIZE = 256
SAMPLE_RATE = 44100
CARRIER_FREQ = 4800
PLOT_UPDATE_INTERVAL_MS = 20
SAMPLES_PER_SYMBOL = 10
MAX_BUFFER_SECONDS = 5.0
MAX_CONSTELLATION_POINTS = 1500
FFT_SIZE = 4096
AUDIO_QUEUE_CHUNKS = 512
LPF_WINDOW = 5
LPF_KERNEL = np.ones(LPF_WINDOW, dtype=float) / LPF_WINDOW

def snapshot_to_array(snapshot):
    if isinstance(snapshot, (list, tuple, deque)):
        if not snapshot:
            return np.array([])
        return np.concatenate(snapshot)

    return np.asarray(snapshot)


def signal_to_symbols(signal):
    signal = snapshot_to_array(signal).astype(float, copy=False)
    signal = signal - np.mean(signal)

    t = np.arange(len(signal)) / SAMPLE_RATE
       
    baseband = signal * np.cos(2 * np.pi * CARRIER_FREQ * t) - 1j * signal * np.sin(2 * np.pi * CARRIER_FREQ * t)
    
    baseband_filtered = convolve(baseband, LPF_KERNEL, mode='same')

    symbols = []
    for i in range(0, len(baseband_filtered), SAMPLES_PER_SYMBOL):
        sample_idx = i
        if sample_idx < len(baseband_filtered):
            symbol = baseband_filtered[sample_idx]
            symbols.append(symbol)

    return np.array(symbols, dtype=complex)
            
def update_plots(signal_snapshot, symbol_snapshot):
    signal_snapshot = snapshot_to_array(signal_snapshot)
    symbol_snapshot = np.asarray(symbol_snapshot, dtype=complex).reshape(-1)

    # Plot 1: Time-domain signals 
    display_duration = anim_state['zoom_duration']  # Adjustable zoom
    display_samples = min(int(display_duration * SAMPLE_RATE), len(signal_snapshot))
    
    ax = axes[0]
    ax.clear()
    if display_samples > 0:
        t = np.arange(display_samples) / SAMPLE_RATE
        ax.plot(t, signal_snapshot[-display_samples:], linewidth=0.5, color='blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Time Domain (TX Signal)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Frequency domain (FFT with windowing)
    ax = axes[1]
    ax.clear()
    if len(signal_snapshot) >= FFT_SIZE:
        windowed_signal = signal_snapshot[-FFT_SIZE:] * np.hanning(FFT_SIZE)
        fft = np.abs(np.fft.rfft(windowed_signal))
        freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        ax.plot(freqs, fft, linewidth=0.5, color='blue')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Frequency Domain (FFT, N={FFT_SIZE})")
        ax.grid(True, alpha=0.3)

    # Plot 3: Constellation
    ax = axes[2]
    ax.clear()
    if len(symbol_snapshot) > 0:
        recent_symbols = symbol_snapshot[-MAX_CONSTELLATION_POINTS:]
        ax.scatter(np.real(recent_symbols), np.imag(recent_symbols), alpha=0.6, s=30, color='blue')
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel("I (In-phase)")
        ax.set_ylabel("Q (Quadrature)")
        ax.set_title(f"Constellation ({len(recent_symbols)} recent symbols)")
        ax.grid(True, alpha=0.3)
        max_lim = 1.5
        ax.set_xlim(-max_lim, max_lim)
        ax.set_ylim(-max_lim, max_lim)

    # Add status overlay (pause/zoom info)
    status_text = f"Zoom: {anim_state['zoom_duration']:.3f}s"
    if anim_state['paused']:
        status_text = "⏸ PAUSED | " + status_text
    status_artist.set_text(status_text)

    return axes.flat

fig, axes = plt.subplots(1, 3, figsize=(14, 8))
fig.suptitle("Transmitter Live View", fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)

status_artist = fig.text(
    0.02,
    0.98,
    "",
    transform=fig.transFigure,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
)
fig.text(
    0.02,
    0.93,
    "[SPACE] Pause | [+/-] Zoom | [0] Reset",
    transform=fig.transFigure,
    fontsize=9,
    verticalalignment='top',
    color='gray',
)

app_state = {'running': True}
audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_CHUNKS)


def on_audio(indata, _frames, _time, _status):
    signal_chunk = indata[:, 0].copy()
    try:
        audio_queue.put_nowait(signal_chunk)
    except queue.Full:
        # Drop oldest chunk so the display stays near real-time.
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            audio_queue.put_nowait(signal_chunk)
        except queue.Full:
            pass


def on_close(_event):
    app_state['running'] = False
    
def on_key_press(event):
    """Handle keyboard pause/zoom controls."""
    if event.key == ' ':
        # Toggle pause
        anim_state['paused'] = not anim_state['paused']
    elif event.key == '+' or event.key == '=':
        # Zoom in (decrease time window)
        anim_state['zoom_duration'] = max(0.001, anim_state['zoom_duration'] * 0.8)
    elif event.key == '-' or event.key == '_':
        # Zoom out (increase time window)
        anim_state['zoom_duration'] = min(5.0, anim_state['zoom_duration'] * 1.25)
    elif event.key == '0':
        # Reset to default
        anim_state['zoom_duration'] = 1.0
                

anim_state = {'paused': False, 'zoom_duration': 1.0}
fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('close_event', on_close)

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, callback=on_audio):
    max_signal_chunks = max(1, int((MAX_BUFFER_SECONDS * SAMPLE_RATE) / CHUNK_SIZE))
    signal_snapshot = deque(maxlen=max_signal_chunks)
    symbol_snapshot = deque(maxlen=MAX_CONSTELLATION_POINTS)
          
    while app_state['running'] and plt.fignum_exists(fig.number):
        try:
            # Drain all captured chunks so ingest rate is independent of plot FPS.
            while True:
                try:
                    signal_chunk = audio_queue.get_nowait()
                except queue.Empty:
                    break

                if not anim_state['paused']:
                    signal_snapshot.append(signal_chunk)
                    new_symbols = signal_to_symbols(signal_chunk)
                    symbol_snapshot.extend(new_symbols)
            
            update_plots(signal_snapshot, symbol_snapshot)
            fig.canvas.draw_idle()
            plt.pause(PLOT_UPDATE_INTERVAL_MS / 1000.0)
        except KeyboardInterrupt:
            print("Exiting...")
            break

print("Exiting...")
        
