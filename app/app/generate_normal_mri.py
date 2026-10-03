"""
Generate a realistic-looking normal healthy spinal cord MRI image
for testing the Normal prediction in the app.
"""
import cv2
import numpy as np
import os

def generate_normal_mri(output_path: str, size: int = 256):
    np.random.seed(42)
    img = np.zeros((size, size), dtype=np.uint8)

    # ── dark background ────────────────────────────────────────────────
    bg = np.random.randint(2, 12, (size, size), dtype=np.uint8)
    img = bg.copy()

    # ── vertebral body regions (dark-ish rectangular areas) ───────────
    cx = size // 2
    vert_positions = np.linspace(30, size - 30, 12).astype(int)
    for yc in vert_positions:
        vw, vh = np.random.randint(55, 70), np.random.randint(14, 20)
        x1, y1 = cx - vw // 2, yc - vh // 2
        x2, y2 = cx + vw // 2, yc + vh // 2
        vert_val = np.random.randint(30, 55)
        cv2.rectangle(img, (x1, y1), (x2, y2), int(vert_val), -1)
        # slight noise on vertebra
        roi = img[y1:y2, x1:x2].astype(np.float32)
        roi += np.random.normal(0, 4, roi.shape).astype(np.float32)
        img[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

    # ── spinal canal (bright CSF stripe) ──────────────────────────────
    canal_w = np.random.randint(10, 15)
    for y in range(20, size - 20):
        xc = cx + np.random.randint(-1, 2)   # slight waviness
        x1 = xc - canal_w // 2
        x2 = xc + canal_w // 2
        # CSF is very bright on T2
        csf_val = np.random.randint(200, 235)
        img[y, x1:x2] = csf_val

    # ── spinal cord (slightly darker stripe inside canal) ─────────────
    cord_w = np.random.randint(5, 8)
    for y in range(25, size - 25):
        xc = cx + np.random.randint(-1, 2)
        x1 = xc - cord_w // 2
        x2 = xc + cord_w // 2
        # Cord is intermediate intensity on T2 (darker than CSF)
        cord_val = np.random.randint(90, 125)
        img[y, x1:x2] = cord_val
        # natural cord texture
        noise_row = np.random.normal(0, 5, x2 - x1).astype(np.float32)
        row = img[y, x1:x2].astype(np.float32) + noise_row
        img[y, x1:x2] = np.clip(row, 60, 150).astype(np.uint8)

    # ── posterior elements / soft tissue ──────────────────────────────
    for y in range(0, size):
        # anterior soft tissue
        x_ant = cx - 45 + np.random.randint(-3, 4)
        img[y, max(0, x_ant - 10):x_ant] = np.random.randint(18, 35, 10)
        # posterior soft tissue
        x_post = cx + 40 + np.random.randint(-3, 4)
        img[y, x_post:min(size, x_post + 10)] = np.random.randint(18, 35, 10)

    # ── slight overall Gaussian blur (scanner PSF) ─────────────────────
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # ── convert to 3-channel grayscale PNG ────────────────────────────
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── add minimal annotation text ───────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bgr, "SAGITTAL T2", (6, 18), font, 0.45, (200, 200, 200), 1)
    cv2.putText(img_bgr, "NORMAL", (6, size - 8), font, 0.45, (80, 200, 80), 1)

    cv2.imwrite(output_path, img_bgr)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "normal_healthy_spinal_mri.png")
    generate_normal_mri(out, size=320)
    print("Done! Upload this image to the app — it should predict as Normal.")
