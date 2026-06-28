import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def create_brain_phantom(size=64):
    """
    Generates a synthetic brain-like phantom using ellipses.
    """
    phantom = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    
    # Outer skull
    mask = ((y - cy)**2 / (0.45 * size)**2 + (x - cx)**2 / (0.35 * size)**2) <= 1.0
    phantom[mask] = 0.6
    
    # Inner structures (ventricles, gray/white matter)
    mask_vent1 = ((y - (cy - 5))**2 / (0.15 * size)**2 + (x - (cx - 8))**2 / (0.08 * size)**2) <= 1.0
    phantom[mask_vent1] = 0.1
    
    mask_vent2 = ((y - (cy - 5))**2 / (0.15 * size)**2 + (x - (cx + 8))**2 / (0.08 * size)**2) <= 1.0
    phantom[mask_vent2] = 0.1
    
    mask_lesion = ((y - (cy + 12))**2 / (0.06 * size)**2 + (x - cx)**2 / (0.12 * size)**2) <= 1.0
    phantom[mask_lesion] = 0.95
    
    return phantom

def simulate_kspace_acquisition(image, noise_sigma=0.05):
    """
    Computes complex k-space representation of the image and adds noise.
    """
    kspace = np.fft.fftshift(np.fft.fft2(image))
    noise = noise_sigma * (np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape)) / np.sqrt(2)
    return kspace + noise

def apply_undersampling_mask(kspace, acceleration=4):
    """
    Applies a Cartesian 1D phase-encode undersampling mask (keeps central 1/acceleration lines).
    """
    h, w = kspace.shape
    mask = np.zeros((h, w), dtype=bool)
    # Keep center lines
    center_width = max(1, w // (acceleration * 2))
    mask[:, w//2 - center_width : w//2 + center_width] = True
    # Keep some random outer lines
    n_random = max(1, w // acceleration - center_width * 2)
    random_cols = np.random.choice(w, size=n_random, replace=False)
    mask[:, random_cols] = True
    
    kspace_us = kspace * mask
    # Inverse FFT to get naive reconstruction
    img_naive = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_us)))
    return kspace_us, img_naive, mask

def psnr_metric(gt, pred):
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return peak_signal_noise_ratio(gt_norm, pred_norm, data_range=1.0)

def ssim_metric(gt, pred):
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return structural_similarity(gt_norm, pred_norm, data_range=1.0)
