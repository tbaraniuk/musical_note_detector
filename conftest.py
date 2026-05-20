import pytest
import numpy as np

from core.audio import generate_note, generate_training_samples, get_mel_spectrum
from core.model import GaussianModel
from core.shared import NOTE_FREQS, N_FFT, mel_filterbank, N_FEATURES


@pytest.fixture
def sample_signal():
    return generate_note(440.0)

@pytest.fixture
def small_model():
    notes = {'C4': 261.63, 'A4': 440.00, 'G4': 392.00}
    X, y = [], []
    for note_name, freq in notes.items():
        samples = generate_training_samples(freq, n_samples=300)
        spectrums = [get_mel_spectrum(s, n_fft=N_FFT, filterbank=mel_filterbank) for s in samples]
        X.extend(spectrums)
        y.extend([note_name] * len(samples))
    model = GaussianModel(list(notes.keys()), num_features=N_FEATURES)
    model.fit(X, y)
    return model, notes
    