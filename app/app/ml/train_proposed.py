"""Fast In-Memory Training for NeuroCordResAttnNet (Proposed Model)."""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.append(str(Path(__file__).parent))
from model import NeuroCordResAttnNet, AttentionUNetWithClassifier, UNetWithClassifier
from utils import CombinedLoss, dice_coefficient, iou_score, classification_accuracy

CLASS_NAMES = ["Tumor", "MS", "Injury", "Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def load_dataset_in_memory(data_dir: Path, img_size: int = 128):
    images = []
    masks = []
    labels = []
    for cls_name in CLASS_NAMES:
        img_dir = data_dir / cls_name / "images"
        mask_dir = data_dir / cls_name / "masks"
        if not img_dir.exists():
            continue
        cls_idx = CLASS_TO_IDX[cls_name]
        for img_path in sorted(img_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
            
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros((img_size, img_size), dtype=np.float32)
            
            images.append(img)
            masks.append(mask)
            labels.append(cls_idx)
    
    x_tensor = torch.from_numpy(np.array(images)).unsqueeze(1)  # (N, 1, H, W)
    m_tensor = torch.from_numpy(np.array(masks)).unsqueeze(1)   # (N, 1, H, W)
    y_tensor = torch.tensor(labels, dtype=torch.long)           # (N,)
    return x_tensor, m_tensor, y_tensor

def train_proposed():
    data_dir = Path(__file__).parent / "data"
    x, m, y = load_dataset_in_memory(data_dir, img_size=128)
    n = len(y)
    print(f"[DATA] Preloaded {n} samples into memory across {len(CLASS_NAMES)} classes: {CLASS_NAMES}", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] Training on {device}", flush=True)

    # Train NeuroCordResAttnNet (Proposed Model)
    model = NeuroCordResAttnNet(in_channels=1, num_classes=4, base=16).to(device)
    optim = Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    loss_fn = CombinedLoss(seg_weight=1.0, cls_weight=1.0)
    
    batch_size = 16
    epochs = 8
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, n, batch_size):
            indices = perm[i : i + batch_size]
            bx = x[indices].to(device)
            bm = m[indices].to(device)
            by = y[indices].to(device)
            
            optim.zero_grad()
            seg_logits, cls_logits = model(bx)
            loss = loss_fn(seg_logits, cls_logits, bm, by)
            loss.backward()
            optim.step()
            
            total_loss += float(loss.item())
            num_batches += 1
        
        if epoch % 2 == 0 or epoch == epochs:
            model.eval()
            dices, ious, accs = [], [], []
            with torch.no_grad():
                for j in range(0, n, batch_size):
                    bx = x[j : j + batch_size].to(device)
                    bm = m[j : j + batch_size].to(device)
                    by = y[j : j + batch_size].to(device)
                    seg, cls = model(bx)
                    dices.append(dice_coefficient(seg, bm))
                    ious.append(iou_score(seg, bm))
                    accs.append(classification_accuracy(cls, by))
            d = sum(dices) / len(dices)
            i_score = sum(ious) / len(ious)
            a = sum(accs) / len(accs)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/max(num_batches,1):.4f} | Dice: {d:.3f} | IoU: {i_score:.3f} | Acc: {a*100:.1f}%", flush=True)

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    proposed_ckpt = ckpt_dir / "best_proposed.pt"
    main_ckpt = ckpt_dir / "best.pt"
    
    state = {
        "model": model.state_dict(),
        "architecture": "NeuroCord-ResAttnNet (Proposed)",
        "base": 16,
        "classes": CLASS_NAMES,
        "dice": float(d),
        "iou": float(i_score),
        "acc": float(a),
    }
    torch.save(state, str(proposed_ckpt))
    torch.save(state, str(main_ckpt))
    print(f"\n[SAVED] Successfully saved proposed checkpoint to {proposed_ckpt} and {main_ckpt}", flush=True)

    # Verify predictions on all 4 classes
    model.eval()
    print("\n--- Model Verification on All 4 Classes ---", flush=True)
    with torch.no_grad():
        for cls_name in CLASS_NAMES:
            cls_idx = CLASS_TO_IDX[cls_name]
            mask_cls = (y == cls_idx)
            cls_x = x[mask_cls]
            correct = 0
            total = len(cls_x)
            all_probs = []
            for k in range(0, total, 16):
                bx = cls_x[k : k + 16].to(device)
                seg, cls = model(bx)
                preds = cls.argmax(dim=1)
                correct += (preds == cls_idx).sum().item()
                p = torch.softmax(cls, dim=1).cpu().numpy()
                all_probs.append(p)
            
            acc_pct = (correct / total) * 100.0 if total > 0 else 0.0
            probs = np.concatenate(all_probs, axis=0).mean(axis=0) if all_probs else np.zeros(4)
            probs_str = ", ".join([f"{CLASS_NAMES[k]}: {probs[k]*100:.1f}%" for k in range(4)])
            print(f"  Class: {cls_name:<8} | Accuracy: {acc_pct:.1f}% ({correct}/{total}) | Mean Probs: [{probs_str}]", flush=True)

if __name__ == "__main__":
    train_proposed()
