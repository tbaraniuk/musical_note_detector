import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core.audio import generate_note, get_mel_spectrum
from core.shared import get_model, NOTE_FREQS, N_FFT, SAMPLE_RATE, mel_filterbank


st.title("Bayesian Updating")
st.text("Each note you play updates the Dirichlet prior in real time.")

model = get_model()

col_reset, col_info = st.columns([1, 4])
with col_reset:
    if st.button("Reset prior"):
        model.alpha = np.ones(model.num_classes)
        st.session_state["history"] = []
        st.session_state["last_result"] = {}
        
with col_info:
    st.caption(f"Total observations: {int(model.alpha.sum() - model.num_classes)}")
    
if "history" not in st.session_state:
    st.session_state["history"] = []
    
st.write("Play a note:")
cols = st.columns(12)
for col, note in zip(cols, NOTE_FREQS.keys()):
    if col.button(note, key=f"bay_{note}"):
        signal = generate_note(NOTE_FREQS[note])
        features = get_mel_spectrum(signal, n_fft=N_FFT, filterbank=mel_filterbank)
        prediction = model.predict(features)[0]
        model.update(prediction)
        st.session_state["history"].append((note, prediction))
        st.session_state["last_result"] = {
            "note": note,
            "prediction": prediction,
            "signal": signal
        }

if ("last_result" in st.session_state) and ("signal" in st.session_state["last_result"]):
    r = st.session_state["last_result"]
    signal = r["signal"]
    st.audio(signal.astype(np.float32), sample_rate=SAMPLE_RATE)
    correct = r["note"] == r["prediction"]
    st.write(f"Played **{r['note']}** → predicted **{r['prediction']}** {'✓' if correct else '✗'}")
        
fig, ax = plt.subplots()
counts = model.alpha - 1
ax.bar(model.classes, counts, color="steelblue")
ax.set_title("Dirichlet α counts (times each note was detected)")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.3, axis="y")
st.pyplot(fig)

if st.session_state["history"]:
    st.write("Detection history")
    cols = st.columns([1, 1, 1])
    cols[0].write("**#**")
    cols[1].write("**Played**")
    cols[2].write("**Predicted**")
    for i, (played, predicted) in enumerate(reversed(st.session_state["history"][-10:])):
        correct = played == predicted
        cols[0].write(len(st.session_state["history"]) - i)
        cols[1].write(played)
        cols[2].write(f"{'✓' if correct else '✗'} {predicted}")