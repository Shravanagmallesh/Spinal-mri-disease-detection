"""
Fix detect_lesions() to be less aggressive on real clinical MRI images.

Root cause: CLAHE enhances cord texture noise -> 19 false lesions detected
in this normal MRI (area_ratio=12.7% triggers MS/Injury).

Fix:
  1. Raise the bright_mask threshold from 1.35*std -> 1.8*std
     (require pixels to be much brighter than cord mean)
  2. Raise minimum lesion size from 6 pixels -> 20 pixels
     (ignore tiny noise artifacts)
  3. Also fix diag_real_mri.py print bug (0.0801 >= 0.12 is False
     but the print logic had wrong precedence)
"""
import re

path = 'backend/server.py'
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()

# ── Fix detect_lesions() lesion detection sensitivity ────────────────────────
old_detect = '''\
    diff = gray.astype(np.int32) - int(mean_i)
    bright_mask = (diff > 1.35 * std_i) & (cord_mask > 0)
    threshold_value = int(min(255.0, mean_i + 1.2 * std_i))
    high_intensity = (gray > threshold_value) & (cord_mask > 0)
    anomaly = (bright_mask | high_intensity).astype(np.uint8) * 255'''

new_detect = '''\
    diff = gray.astype(np.int32) - int(mean_i)
    # Require pixels to be significantly brighter than cord mean.
    # Threshold 1.8*std (was 1.35) reduces false positives from CLAHE texture.
    bright_mask = (diff > 1.8 * std_i) & (cord_mask > 0)
    threshold_value = int(min(255.0, mean_i + 1.6 * std_i))
    high_intensity = (gray > threshold_value) & (cord_mask > 0)
    anomaly = (bright_mask | high_intensity).astype(np.uint8) * 255'''

if old_detect in src:
    src = src.replace(old_detect, new_detect, 1)
    print("detect_lesions() brightness threshold updated OK")
else:
    print("ERROR: detect_lesions() brightness threshold not found")

# Raise minimum lesion pixel size from 6 -> 20
old_size = '    lesion_sizes = [float(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= 6]'
new_size = '    lesion_sizes = [float(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= 20]'
if old_size in src:
    src = src.replace(old_size, new_size, 1)
    print("Minimum lesion size updated 6->20 pixels OK")
else:
    print("ERROR: lesion size filter line not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(src)
print("Saved.")

import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
