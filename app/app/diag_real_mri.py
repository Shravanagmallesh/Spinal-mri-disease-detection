"""Final diagnostic using UPDATED server.py thresholds exactly."""
import sys, cv2, numpy as np
sys.path.insert(0, 'backend')

IMG_PATH = r'C:/Users/kukku/.gemini/antigravity/brain/08f0243a-ea06-4ee6-bf94-f0c6041d03ad/.user_uploaded/media_1790692687431.jpg'
IMG_SIZE = 224

# ── preprocess ──────────────────────────────────────────────────────────────
with open(IMG_PATH, 'rb') as f:
    raw = f.read()
arr = np.frombuffer(raw, dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
gray = clahe.apply(img)

# ── segment_spinal_cord ─────────────────────────────────────────────────────
h, w = gray.shape
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
central = np.zeros_like(otsu)
cx0, cx1 = int(w * 0.28), int(w * 0.72)
central[:, cx0:cx1] = 255
cord = cv2.bitwise_and(otsu, central)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cord = cv2.morphologyEx(cord, cv2.MORPH_OPEN, kernel, iterations=1)
cord = cv2.morphologyEx(cord, cv2.MORPH_CLOSE, kernel, iterations=3)
num, labels, stats, _ = cv2.connectedComponentsWithStats(cord, connectivity=8)
if num > 1:
    larg_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cord = (labels == larg_idx).astype(np.uint8) * 255
cord_mask = cord

# ── detect_lesions (UPDATED thresholds: 1.8*std, 1.6*std, min_size=20) ──────
cord_pixels = gray[cord_mask > 0]
cord_area = int(cord_mask.sum() / 255)
mean_i = float(cord_pixels.mean())
std_i  = float(cord_pixels.std() + 1e-6)
diff        = gray.astype(np.int32) - int(mean_i)
bright_mask = (diff > 1.8 * std_i) & (cord_mask > 0)          # UPDATED: was 1.35
threshold_v = int(min(255.0, mean_i + 1.6 * std_i))            # UPDATED: was 1.2
high_int    = (gray > threshold_v) & (cord_mask > 0)
anomaly     = (bright_mask | high_int).astype(np.uint8) * 255
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN,  k3, iterations=1)
anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, k5, iterations=2)
anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, k7, iterations=1)
anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN,  k3, iterations=1)
ncc, lab2, st2, _ = cv2.connectedComponentsWithStats(anomaly, connectivity=8)
lesion_count = max(0, ncc - 1)
lesion_sizes = [float(st2[i, cv2.CC_STAT_AREA]) for i in range(1, ncc)
                if st2[i, cv2.CC_STAT_AREA] >= 20]              # UPDATED: was 6
total_area   = float(sum(lesion_sizes))
largest_area = float(max(lesion_sizes, default=0.0))
area_ratio   = total_area / max(cord_area, 1)
largest_ratio= largest_area / max(cord_area, 1)

# ── compute_additional_features (UPDATED: local-segment deformity) ──────────
elong = 1.0
contours_l, _ = cv2.findContours(anomaly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours_l:
    lc = max(contours_l, key=cv2.contourArea)
    x, y, lw, lh = cv2.boundingRect(lc)
    elong = max(lw, lh) / max(min(lw, lh), 1)

slice_h = max(8, h // 16)
defects = []
for y0 in range(0, h - slice_h, slice_h // 2):
    y1    = min(y0 + slice_h, h)
    strip = cord_mask[y0:y1, :]
    if strip.sum() < 50:
        continue
    cnts, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    lc2  = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(lc2)
    if area < 10:
        continue
    hull = cv2.convexHull(lc2)
    ha   = cv2.contourArea(hull) + 1e-6
    defects.append(float(1.0 - area / ha))

deform = 0.0
if defects:
    defects.sort()
    idx = int(len(defects) * 0.75)
    deform = defects[idx]

healthy_mask = (cord_mask > 0) & (anomaly == 0)
contrast = 0.0
if anomaly.sum() and healthy_mask.sum():
    lm = float(gray[anomaly > 0].mean())
    hm = float(gray[healthy_mask].mean()) + 1e-6
    contrast = abs(lm - hm) / hm

print("=== FINAL pipeline features (all fixes applied) ===")
print(f"lesion_count       : {lesion_count}")
print(f"total_area_ratio   : {area_ratio:.4f}  ({area_ratio*100:.2f}%)")
print(f"largest_ratio      : {largest_ratio:.4f}  ({largest_ratio*100:.2f}%)")
print(f"lesion_elongation  : {elong:.2f}")
print(f"cord_deformity     : {deform:.4f}  (local-segment method)")
print(f"intensity_contrast : {contrast:.4f}")
print()
print("=== classify() rule evaluation ===")
r1 = deform >= 0.12 or (elong >= 2.5 and area_ratio >= 0.020 and deform >= 0.06)
r2 = lesion_count == 0 or area_ratio < 0.005
r3_large = largest_ratio >= 0.06
r3_dom   = area_ratio >= 0.04 and lesion_count <= 2
r3 = r3_large or r3_dom
r4 = lesion_count >= 2 and largest_ratio < 0.06
print(f"Rule 1 INJURY : {r1}  (deform={deform:.4f}>=0.12 is {deform>=0.12})")
print(f"Rule 2 NORMAL : {r2}  (count={lesion_count}==0 is {lesion_count==0}, ratio={area_ratio:.4f}<0.005 is {area_ratio<0.005})")
print(f"Rule 3 TUMOR  : {r3}  (largest={largest_ratio:.4f}>=0.06 is {r3_large}, dom={r3_dom})")
print(f"Rule 4 MS     : {r4}  (count={lesion_count}>=2 and largest<0.06)")
result = "INJURY" if r1 else ("NORMAL" if r2 else ("TUMOR" if r3 else ("MS" if r4 else "FALLBACK")))
print()
print(">>> FINAL PREDICTED CLASS:", result)
