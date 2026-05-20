import streamlit as st
import numpy as np

from core.audio import generate_training_samples, get_mel_spectrum, get_mel_filterbank
from core.model import GaussianModel


NOTE_FREQS = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88
}

SAMPLE_RATE = 44100
N_FFT = 2048
N_FEATURES = 40

BIN_FREQUENCIES = np.fft.rfftfreq(2048, d=1/SAMPLE_RATE)  
USEFUL_BINS = np.where((BIN_FREQUENCIES >= 80) & (BIN_FREQUENCIES <= 2000))[0] 

mel_filterbank = get_mel_filterbank(SAMPLE_RATE, n_fft=N_FFT, n_mels=N_FEATURES, fmin=20, fmax=2000)


@st.cache_data(show_spinner="Training model on synthetic data...")
def build_training_data():
    X, y = [], []
    for note_name, freq in NOTE_FREQS.items():
        training_samples = generate_training_samples(freq, 100)
        spectrums = [get_mel_spectrum(sample, n_fft=N_FFT, filterbank=mel_filterbank) for sample in training_samples]
        X.extend(spectrums)
        y.extend([note_name] * len(training_samples))
    return np.array(X), np.array(y)

def get_model():
    if "model" not in st.session_state:
        X, y = build_training_data()
        model = GaussianModel(classes=list(NOTE_FREQS.keys()), num_features=N_FEATURES)
        model.fit(X, y)
        st.session_state["model"] = model
    return st.session_state["model"]
