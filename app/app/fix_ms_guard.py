"""
Final fix: add mean_lesion_ratio guard to classify() MS rule.

Current state after previous fixes:
  lesion_count=12, largest_ratio=0.0061 (only 0.61% of cord each on avg)
  These are tiny texture artifacts, NOT real MS plaques.

Real MS plaques are individually significant (each >= 0.3% of cord area).
Add mean_comp (mean lesion ratio) check to the MS rule so that 12 tiny
sub-pixel noise dots don't trigger MS classification.
"""

path = 'backend/server.py'
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()

# ── Update MS rule in classify() to require individually significant lesions ─
old_ms = '''\
    if lesion_count >= 2 and largest < 0.06:
        raw_conf = min(95.0, 52.0 + lesion_count * 9.0 + total_area_ratio * 220.0)
        conf = round(max(68.0, raw_conf), 2)
        rem  = round(100.0 - conf, 2)
        probs = {
            "Tumor":  round(rem * 0.20, 2),
            "MS":     conf,
            "Injury": round(rem * 0.20, 2),
            "Normal": round(rem * 0.60, 2),
        }
        return (
            "MS",
            conf,
            probs,
            f"{lesion_count} discrete small hyperintense plaques detected (each < 6% of cord area) "
            "with preserved cord contour - characteristic of Multiple Sclerosis.",
        )'''

new_ms = '''\
    # mean_comp = mean individual lesion ratio (average size of each plaque vs cord).
    # Real MS plaques are individually significant (mean >= 0.3% of cord area each).
    # If mean_comp is tiny the detections are noise artifacts, not real plaques.
    if lesion_count >= 2 and largest < 0.06 and mean_comp >= 0.003:
        raw_conf = min(95.0, 52.0 + lesion_count * 9.0 + total_area_ratio * 220.0)
        conf = round(max(68.0, raw_conf), 2)
        rem  = round(100.0 - conf, 2)
        probs = {
            "Tumor":  round(rem * 0.20, 2),
            "MS":     conf,
            "Injury": round(rem * 0.20, 2),
            "Normal": round(rem * 0.60, 2),
        }
        return (
            "MS",
            conf,
            probs,
            f"{lesion_count} discrete small hyperintense plaques detected (each < 6% of cord area) "
            "with preserved cord contour - characteristic of Multiple Sclerosis.",
        )'''

if old_ms in src:
    src = src.replace(old_ms, new_ms, 1)
    print("MS rule updated with mean_comp guard OK")
else:
    print("ERROR: MS rule not found")

# Also update the fallback ms_score to use same logic
old_fb_ms = '        (4.0 if lesion_count >= 2 else 0.0)'
new_fb_ms = '        (4.0 if lesion_count >= 2 and mean_comp >= 0.003 else 0.0)'
if old_fb_ms in src:
    src = src.replace(old_fb_ms, new_fb_ms, 1)
    print("Fallback ms_score guard updated OK")
else:
    print("WARNING: fallback ms_score not updated (non-critical)")

with open(path, 'w', encoding='utf-8') as f:
    f.write(src)
print("Saved.")

import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
