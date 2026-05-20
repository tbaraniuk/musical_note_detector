import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core.audio import generate_note, get_fft_spectrum
from core.shared import BIN_FREQUENCIES, NOTE_FREQS, SAMPLE_RATE, N_FFT, USEFUL_BINS


st.title("Notes Explorer")
st.text("Listen to each note and see its waveform and spectrum analysis")

selected = st.segmented_control(
    "Pick a note",
    options=list(NOTE_FREQS.keys()),
    default="A4"
)

if selected:
    signal = generate_note(NOTE_FREQS[selected])
    
    st.audio(signal.astype(np.float32), sample_rate=SAMPLE_RATE)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        ax.plot(signal[:500])
        ax.set_title(f"Waveform: {selected} (first 500 samples)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        spectrum = get_fft_spectrum(signal, n_fft=N_FFT)[USEFUL_BINS]
        ax.plot(BIN_FREQUENCIES[USEFUL_BINS], spectrum)
        ax.set_title(f"FFT spectrum: {selected}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Log magnitude")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    st.caption("Expected harmonic peaks")
    freq = NOTE_FREQS[selected]
    harmonics = [freq * k for k in range(1, 7) if freq * k <= 5000]
    st.write(" · ".join([f"{h:.1f} Hz" for h in harmonics]))
