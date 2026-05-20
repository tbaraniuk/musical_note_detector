import numpy as np
import pytest


def test_model_covariance_finite(small_model):
    model, _ = small_model
    for i in range(model.num_classes):
        assert np.isfinite(model.sigma[i]).all(), \
            f"Sigma for class {i} contains nan/inf"
            
def test_model_covariance_positive_definite(small_model):
    model, _ = small_model
    for i, name in enumerate(model.classes):
        eigenvalues = np.linalg.eigvalsh(model.sigma[i])
        assert (eigenvalues > 0).all(), \
            f"Sigma for class {name} is not positive definite (min eigenvalue={eigenvalues.min():.2e})"

def test_dirichlet_update(small_model):
    model, _ = small_model
    alpha_before = model.alpha.copy()
    model.update("A4")
    idx = model.classes == "A4"
    assert model.alpha[idx] == alpha_before[idx] + 1
    assert model.alpha.sum() == alpha_before.sum() + 1
