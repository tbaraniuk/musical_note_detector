import numpy as np
import pytest

from core.audio import generate_note, generate_training_samples


def test_generate_note_shape():
    signal = generate_note(440.0, duration=0.5, sr=44100)
    assert len(signal) == 44100 * 0.5
    
def test_generate_note_normalized(sample_signal):
    assert np.max(np.abs(sample_signal)) <= 1.0, "Signal must be in [-1, 1]"
    
def test_generate_not_not_silent(sample_signal):
    assert np.std(sample_signal) > 0.01, "Signal should have meaningful magnitude"
    
def test_different_notes_sound_different():
    sig_c = generate_note(261.63)
    sig_a = generate_note(440.00)
    
    assert not np.allclose(sig_c, sig_a), "C4 and A4 should produce different signals"
    
def test_training_samples_count():
    samples = generate_training_samples(440.0, n_samples=100)
    assert len(samples) == 100
    