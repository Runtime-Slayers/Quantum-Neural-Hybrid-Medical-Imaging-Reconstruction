import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from quantum_imaging import (
    QuantumResidualAttentionReconstructor,
    create_brain_phantom,
    simulate_kspace_acquisition,
    apply_undersampling_mask,
    psnr_metric,
    ssim_metric
)
from quantum_imaging.models import ClassicalOnlyReconstructor

def extract_patches(img, size=16):
    h, w = img.shape
    patches = []
    for y in range(0, h, size):
        for x in range(0, w, size):
            patches.append(img[y:y+size, x:x+size])
    return np.array(patches)

def reconstruct_from_patches(patches, img_shape, size=16):
    h, w = img_shape
    reconstructed = np.zeros(img_shape)
    idx = 0
    for y in range(0, h, size):
        for x in range(0, w, size):
            reconstructed[y:y+size, x:x+size] = patches[idx]
            idx += 1
    return reconstructed

def main():
    parser = argparse.ArgumentParser(description="Hybrid QML Medical Imaging Reconstruction Simulation CLI")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--acceleration", type=int, default=4, help="MRI acceleration factor")
    parser.add_argument("--noise-level", type=float, default=0.08, help="k-space noise sigma")
    parser.add_argument("--output-json", type=str, default="output/reconstruction_results.json", help="Path to save output results")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()
    
    logger.info("Initializing Quantum-Classical Medical Imaging Pipeline...")
    
    # 1. Create Brain Phantom (size 64x64) and simulate acquisition
    phantom = create_brain_phantom(64)
    kspace_noisy = simulate_kspace_acquisition(phantom, noise_sigma=args.noise_level)
    kspace_us, img_naive, mask = apply_undersampling_mask(kspace_noisy, acceleration=args.acceleration)
    
    # 2. Extract patches (16x16 size)
    gt_patches = extract_patches(phantom, 16)       # Shape: (16, 16, 16)
    naive_patches = extract_patches(img_naive, 16)   # Shape: (16, 16, 16)
    
    # Convert to PyTorch Tensors
    X_train = torch.tensor(naive_patches, dtype=torch.float32).unsqueeze(1) # Shape: (16, 1, 16, 16)
    y_train = torch.tensor(gt_patches, dtype=torch.float32).unsqueeze(1)    # Shape: (16, 1, 16, 16)
    
    # 3. Instantiate Models
    q_reconstructor = QuantumResidualAttentionReconstructor(n_layers=2)
    c_reconstructor = ClassicalOnlyReconstructor()
    
    # 4. Joint Training Loop for Quantum-Classical Reconstructor
    logger.info("Training Hybrid Quantum-Classical model...")
    q_optimizer = optim.Adam(q_reconstructor.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    q_losses = []
    for epoch in range(1, args.epochs + 1):
        q_reconstructor.train()
        q_optimizer.zero_grad()
        outputs = q_reconstructor(X_train)
        loss = criterion(outputs, y_train)
        # Add TV regularization
        tv_loss = torch.sum(torch.abs(outputs[:, :, 1:, :] - outputs[:, :, :-1, :])) +                   torch.sum(torch.abs(outputs[:, :, :, 1:] - outputs[:, :, :, :-1]))
        total_loss = loss + 0.01 * tv_loss
        
        total_loss.backward()
        q_optimizer.step()
        q_losses.append(total_loss.item())
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:2d}/{args.epochs:2d} | Quantum Loss: {total_loss.item():.4f}")
            
    # 5. Training Loop for Classical-Only Reconstructor
    logger.info("Training Classical-Only model...")
    c_optimizer = optim.Adam(c_reconstructor.parameters(), lr=args.lr)
    c_losses = []
    for epoch in range(1, args.epochs + 1):
        c_reconstructor.train()
        c_optimizer.zero_grad()
        outputs = c_reconstructor(X_train)
        loss = criterion(outputs, y_train)
        tv_loss = torch.sum(torch.abs(outputs[:, :, 1:, :] - outputs[:, :, :-1, :])) +                   torch.sum(torch.abs(outputs[:, :, :, 1:] - outputs[:, :, :, :-1]))
        total_loss = loss + 0.01 * tv_loss
        
        total_loss.backward()
        c_optimizer.step()
        c_losses.append(total_loss.item())
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:2d}/{args.epochs:2d} | Classical Loss: {total_loss.item():.4f}")
            
    # 6. Evaluation
    q_reconstructor.eval()
    c_reconstructor.eval()
    with torch.no_grad():
        q_recon_patches = q_reconstructor(X_train).squeeze(1).numpy()
        c_recon_patches = c_reconstructor(X_train).squeeze(1).numpy()
        
    img_quantum = reconstruct_from_patches(q_recon_patches, (64, 64), 16)
    img_classical = reconstruct_from_patches(c_recon_patches, (64, 64), 16)
    
    # Calculate Metrics
    psnr_naive = psnr_metric(phantom, img_naive)
    ssim_naive = ssim_metric(phantom, img_naive)
    
    psnr_c = psnr_metric(phantom, img_classical)
    ssim_c = ssim_metric(phantom, img_classical)
    
    psnr_q = psnr_metric(phantom, img_quantum)
    ssim_q = ssim_metric(phantom, img_quantum)
    
    logger.info(f"Naive:     PSNR={psnr_naive:.2f} dB | SSIM={ssim_naive:.4f}")
    logger.info(f"Classical: PSNR={psnr_c:.2f} dB | SSIM={ssim_c:.4f}")
    logger.info(f"Quantum:   PSNR={psnr_q:.2f} dB | SSIM={ssim_q:.4f}")
    
    results = {
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "acceleration": args.acceleration,
            "noise_level": args.noise_level
        },
        "metrics": {
            "naive": {"psnr_db": psnr_naive, "ssim": ssim_naive},
            "classical_only": {"psnr_db": psnr_c, "ssim": ssim_c},
            "quantum_hybrid": {"psnr_db": psnr_q, "ssim": ssim_q}
        },
        "training_curves": {
            "quantum_loss": q_losses,
            "classical_loss": c_losses
        }
    }
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_json}")
    
    if not args.no_plots:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Images
        axes[0, 0].imshow(phantom, cmap='gray')
        axes[0, 0].set_title("Ground Truth Phantom")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_naive, cmap='gray')
        axes[0, 1].set_title(f"Naive Recon\nPSNR={psnr_naive:.1f}dB, SSIM={ssim_naive:.3f}")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_quantum, cmap='gray')
        axes[0, 2].set_title(f"Quantum Hybrid Recon\nPSNR={psnr_q:.1f}dB, SSIM={ssim_q:.3f}")
        axes[0, 2].axis('off')
        
        # Row 2: Loss Curves and Classical
        axes[1, 0].plot(q_losses, label="Quantum-Classical Loss", color="blue")
        axes[1, 0].plot(c_losses, label="Classical-Only Loss", color="red")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Training Loss Convergence")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)
        
        axes[1, 1].imshow(img_classical, cmap='gray')
        axes[1, 1].set_title(f"Classical-Only Recon\nPSNR={psnr_c:.1f}dB, SSIM={ssim_c:.3f}")
        axes[1, 1].axis('off')
        
        # Undersampling Mask
        axes[1, 2].imshow(mask, cmap='gray', aspect='auto')
        axes[1, 2].set_title("k-space Undersampling Mask")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plot_path = args.output_json.replace('.json', '_plots.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Reconstruction comparison plots saved to {plot_path}")

if __name__ == "__main__":
    main()
