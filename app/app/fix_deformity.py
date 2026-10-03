"""
Fix cord_deformity calculation and classify() Normal thresholds.

Root causes found from diagnostic:
  cord_deformity = 0.2662  -> Rule 1 INJURY fires (threshold was 0.05)
  But this is the NATURAL spinal curve (lordosis/kyphosis), NOT real deformity!
  The convex hull of a curved spine is much larger than the cord itself,
  giving a falsely high deformity score.

Fix:
  1. compute_additional_features(): compute deformity on each cord SEGMENT
     (short vertical slices) rather than the whole cord contour.
     A naturally curved cord is locally straight in each segment.
     Only actual breaks/kinks in the cord raise the local deformity score.
  2. classify(): raise Injury deformity threshold from 0.05 -> 0.10
     and add elong guard so elongated-lesion CSF pattern isn't Injury.
"""
import re

path = 'backend/server.py'
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()

# ── 1. Replace compute_additional_features() ─────────────────────────────────
old_fn = '''\
def compute_additional_features(gray: np.ndarray, cord_mask: np.ndarray, lesion_mask: np.ndarray) -> Dict[str, float]:
    features = {
        "cord_deformity": 0.0,
        "lesion_elongation": 0.0,
        "intensity_contrast": 0.0,
    }
    cord_area = int(cord_mask.sum() / 255) + 1

    healthy_mask = (cord_mask > 0) & (lesion_mask == 0)
    if lesion_mask.sum() and healthy_mask.sum():
        lesion_mean = float(gray[lesion_mask > 0].mean())
        healthy_mean = float(gray[healthy_mask].mean()) + 1e-6
        features["intensity_contrast"] = abs(lesion_mean - healthy_mean) / healthy_mean

    contours, _ = cv2.findContours(cord_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_c)
        hull = cv2.convexHull(largest_c)
        hull_area = cv2.contourArea(hull) + 1e-6
        features["cord_deformity"] = float(1.0 - area / hull_area)

    contours_l, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_l:
        largest = max(contours_l, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        features["lesion_elongation"] = max(w, h) / max(min(w, h), 1)

    return features'''

new_fn = '''\
def compute_additional_features(gray: np.ndarray, cord_mask: np.ndarray, lesion_mask: np.ndarray) -> Dict[str, float]:
    """Compute morphological features.

    cord_deformity is calculated using LOCAL segments of the cord
    (short horizontal slices) rather than the whole contour.
    This prevents the natural spinal curve (lordosis/kyphosis) from
    being mistaken for traumatic deformity: a naturally curved cord
    is locally straight in each segment, so its local deformity score
    stays near zero.  Only genuine breaks/kinks produce a high score.
    """
    features = {
        "cord_deformity": 0.0,
        "lesion_elongation": 1.0,
        "intensity_contrast": 0.0,
    }

    # Intensity contrast: lesion vs healthy cord
    healthy_mask = (cord_mask > 0) & (lesion_mask == 0)
    if lesion_mask.sum() and healthy_mask.sum():
        lesion_mean  = float(gray[lesion_mask > 0].mean())
        healthy_mean = float(gray[healthy_mask].mean()) + 1e-6
        features["intensity_contrast"] = abs(lesion_mean - healthy_mean) / healthy_mean

    # LOCAL cord deformity: measure convexity in horizontal slices
    h, w = cord_mask.shape
    slice_h   = max(8, h // 16)   # height of each segment
    defects   = []
    for y0 in range(0, h - slice_h, slice_h // 2):
        y1    = min(y0 + slice_h, h)
        strip = cord_mask[y0:y1, :]
        if strip.sum() < 50:       # skip near-empty rows
            continue
        cnts, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        lc   = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(lc)
        if area < 10:
            continue
        hull = cv2.convexHull(lc)
        ha   = cv2.contourArea(hull) + 1e-6
        defects.append(float(1.0 - area / ha))

    if defects:
        # Use 75th-percentile so occasional noisy slices don't dominate
        defects.sort()
        idx = int(len(defects) * 0.75)
        features["cord_deformity"] = defects[idx]

    # Lesion elongation
    contours_l, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_l:
        largest = max(contours_l, key=cv2.contourArea)
        x, y, lw, lh = cv2.boundingRect(largest)
        features["lesion_elongation"] = max(lw, lh) / max(min(lw, lh), 1)

    return features'''

if old_fn in src:
    src = src.replace(old_fn, new_fn, 1)
    print("compute_additional_features() replaced OK")
else:
    print("ERROR: compute_additional_features() not found — check whitespace/CRLF")

# ── 2. Replace classify() Injury threshold ───────────────────────────────────
# Change deform >= 0.05 to >= 0.12 in Rule 1 of classify()
# so natural spinal curves (deform ~0.05-0.10) don't trigger Injury.
old_inj = '    if deform >= 0.05 or (elong >= 2.5 and total_area_ratio >= 0.010):'
new_inj = '    if deform >= 0.12 or (elong >= 2.5 and total_area_ratio >= 0.020 and deform >= 0.06):'

if old_inj in src:
    src = src.replace(old_inj, new_inj, 1)
    print("Injury threshold updated OK")
else:
    print("ERROR: Injury threshold line not found")

# Also update fallback injury_score threshold
old_fb = '        (5.0 if deform >= 0.05 else 0.0)'
new_fb = '        (5.0 if deform >= 0.12 else 0.0)'
if old_fb in src:
    src = src.replace(old_fb, new_fb, 1)
    print("Fallback injury_score threshold updated OK")
else:
    print("WARNING: fallback injury_score not found (non-critical)")

with open(path, 'w', encoding='utf-8') as f:
    f.write(src)

print("Done.")

# Verify syntax
import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
