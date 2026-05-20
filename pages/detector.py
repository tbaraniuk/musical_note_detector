import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core.audio import generate_note, get_mel_spectrum
from core.shared import NOTE_FREQS, N_FFT, SAMPLE_RATE, get_model, mel_filterbank


st.title("Note Detector")
st.write("Generate a note and see if the model identifies it correctly.")

model = get_model()

noise = st.slider("Noise level", min_value=0.0, max_value=0.2, value=0.02, step=0.01)

st.write("Pick a note to generate:")
cols = st.columns(12)
clicked_note = None
for col, note in zip(cols, NOTE_FREQS.keys()):
    if col.button(note, key=f"det_{note}"):
        clicked_note = note
        
if clicked_note is not None:
    st.session_state["last_detected"] = clicked_note
    

if "last_detected" in st.session_state:
    note = st.session_state["last_detected"]
    freq = NOTE_FREQS[note]

    signal = generate_note(freq)
    signal += np.random.normal(0, noise, len(signal))
    signal = np.clip(signal, -1, 1)

    features = get_mel_spectrum(signal, n_fft=N_FFT, filterbank=mel_filterbank)
    prediction = model.predict(features)[0]
    correct = prediction == note

    st.audio(signal.astype(np.float32), sample_rate=SAMPLE_RATE)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("True note", note)
        st.metric("Predicted", prediction)
        if correct:
            st.success("Correct!")
        else:
            st.error("Wrong!")

    with col2:
        log_posteriors = np.array([
            model._log_likelihood(features, k) + np.log(model.alpha[k] / model.alpha.sum())
            for k in range(model.num_classes)
        ])
        log_posteriors -= log_posteriors.max()
        probs = np.exp(log_posteriors) / np.exp(log_posteriors).sum()

        fig, ax = plt.subplots()
        colors = ["green" if n == prediction else "steelblue" for n in model.classes]
        ax.barh(model.classes, probs, color=colors)
        ax.set_xlabel("Posterior probability")
        ax.set_title("Model confidence per note")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)
