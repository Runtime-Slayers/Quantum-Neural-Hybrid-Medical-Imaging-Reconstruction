import pytest
import torch
import numpy as np
from quantum_imaging.models import QuantumClassicalReconstructor
from quantum_imaging.pipeline import create_brain_phantom, simulate_kspace_acquisition, apply_undersampling_mask

def test_model_forward_pass():
    model = QuantumClassicalReconstructor(n_layers=1)
    x = torch.randn(2, 1, 16, 16) # batch of 2 patches
    y = model(x)
    assert y.shape == (2, 1, 16, 16)
    # Check pixels are in [0, 1] due to Sigmoid
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)

def test_brain_phantom():
    phantom = create_brain_phantom(32)
    assert phantom.shape == (32, 32)
    assert phantom.max() == pytest.approx(0.95)
    assert phantom.min() == 0.0

def test_kspace_simulation():
    img = np.random.rand(16, 16)
    kspace = simulate_kspace_acquisition(img, noise_sigma=0.01)
    assert kspace.shape == (16, 16)
    assert np.iscomplexobj(kspace)
    
    kspace_us, img_naive, mask = apply_undersampling_mask(kspace, acceleration=2)
    assert kspace_us.shape == (16, 16)
    assert img_naive.shape == (16, 16)
    assert mask.shape == (16, 16)
    assert mask.dtype == bool

def test_residual_attention_model():
    from quantum_imaging.models import QuantumResidualAttentionReconstructor
    model = QuantumResidualAttentionReconstructor(n_layers=1)
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 16, 16)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
