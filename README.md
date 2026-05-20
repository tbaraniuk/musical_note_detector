# Musical Note Detector

Simple Streamlit app for exploring musical notes, generating test sounds, and checking how well the model predicts them.

Deployment: https://musicalnotedetector-gcu8xyqv2vkz922ydnrsws.streamlit.app/

## What it does

- Note Explorer: play a note and inspect its waveform and spectrum.
- Note Detector: generate a note, add a little noise, and see the model predict it.
- Bayesian Updating: watch the model update its note counts over time.

## Run locally

This project uses `uv`.

```bash
uv sync
uv run streamlit run app.py
```

## Project structure

- `app.py`: Streamlit navigation and page entry point.
- `pages/notes.py`: note explorer page.
- `pages/detector.py`: generate a note and predict it.
- `pages/bayesian.py`: live Bayesian updating page.
- `core/`: audio, shared constants, and model logic.

## Notes

- Python 3.13 or newer is required.
- The app is built with Streamlit, NumPy, SciPy, Matplotlib.
