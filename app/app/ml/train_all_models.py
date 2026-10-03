"""Quick training & verification script for all 3 spinal cord MRI diagnostic models:
  1. NeuroCordResAttnNet (Proposed Model)
  2. AttentionUNetWithClassifier
  3. UNetWithClassifier
"""
import os
import sys
from pathlib import Path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from data_loader import SpinalCordMRIDataset, CLASS_NAMES
from model import UNetWithClassifier, AttentionUNetWithClassifier, NeuroCordResAttnNet
from utils import CombinedLoss, dice_coefficient, iou_score, classification_accuracy

def train_model(model_cls, name, save_path, epochs=12, lr=1e-3, base=16, img_size=128, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n==========================================")
    print(f" Training {name} on {device}")
    print(f"==========================================")
    
    data_dir = Path(__file__).parent / "data"
    dataset = SpinalCordMRIDataset(data_dir, image_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = model_cls(in_channels=1, num_classes=len(CLASS_NAMES), base=base).to(device)
    optim = Adam(model.parameters(), lr=lr)
    loss_fn = CombinedLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, m, y in loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optim.zero_grad()
            seg, cls = model(x)
            loss = loss_fn(seg, cls, m, y)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
        
        if epoch % 3 == 0 or epoch == epochs:
            model.eval()
            dices, accs = [], []
            with torch.no_grad():
                for x, m, y in loader:
                    x, m, y = x.to(device), m.to(device), y.to(device)
                    seg, cls = model(x)
                    dices.append(dice_coefficient(seg, m))
                    accs.append(classification_accuracy(cls, y))
            avg_d = sum(dices)/len(dices)
            avg_a = sum(accs)/len(accs)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/len(loader):.4f} | Dice: {avg_d:.3f} | Acc: {avg_a:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "architecture": name,
        "base": base,
        "classes": CLASS_NAMES,
    }, save_path)
    print(f"[SUCCESS] Saved {name} checkpoint to {save_path}")

    # Test class-wise predictions
    model.eval()
    print(f"\nEvaluating {name} on individual sample classes:")
    with torch.no_grad():
        for cls_name in CLASS_NAMES:
            img_path = data_dir / cls_name / "images" / f"{cls_name}_000.png"
            if img_path.exists():
                import cv2
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                x = torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
                seg, cls = model(x)
                probs = torch.softmax(cls, dim=1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                pred_name = CLASS_NAMES[pred_idx]
                print(f"  GT: {cls_name:<8} -> Predicted: {pred_name:<8} (Confidence: {probs[pred_idx]*100:.1f}%) Probs: {[round(float(p)*100,1) for p in probs]}")

if __name__ == "__main__":
    import numpy as np
    ckpt_dir = Path(__file__).parent / "checkpoints"
    
    # Train Proposed NeuroCordResAttnNet
    train_model(NeuroCordResAttnNet, "NeuroCord-ResAttnNet (Proposed)", str(ckpt_dir / "best_proposed.pt"), epochs=12, base=16)
    
    # Also save as main best.pt
    import shutil
    shutil.copy(str(ckpt_dir / "best_proposed.pt"), str(ckpt_dir / "best.pt"))
    print("\nCopied best_proposed.pt to best.pt as default model checkpoint.")
