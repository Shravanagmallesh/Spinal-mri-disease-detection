import re

path = r'backend/server.py'
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()

new_classify = r'''def classify(
    lesion_count: int,
    total_area_ratio: float,
    largest: float,
    mean_comp: float,
    contrast: float,
    deform: float,
    elong: float,
    std_i: float,
) -> Tuple[str, float, Dict[str, float], str]:
    """Morphological feature-based 4-class classifier for spinal cord pathology.

    Rules applied in priority order:
      1. Injury  - cord structural break or deformity detected (checked FIRST, even with
                   zero bright lesions, so broken cords are never mislabelled as Normal)
      2. Normal  - no bright regions AND no structural deformity found
      3. Tumor   - one large dominant bright region in the cord
      4. MS      - multiple small discrete bright regions scattered along the cord
    """

    # ------------------------------------------------------------------
    # RULE 1 - INJURY: cord break / deformity
    #   Checked BEFORE lesion count so deformed cords with no bright
    #   lesions are NOT mislabelled as Normal.
    # ------------------------------------------------------------------
    if deform >= 0.05 or (elong >= 2.5 and total_area_ratio >= 0.010):
        raw_conf = min(97.0, 68.0 + deform * 280.0 + elong * 3.0)
        conf = round(max(75.0, raw_conf), 2)
        rem = round(100.0 - conf, 2)
        probs = {
            "Tumor":  round(rem * 0.25, 2),
            "MS":     round(rem * 0.25, 2),
            "Injury": conf,
            "Normal": round(rem * 0.50, 2),
        }
        return (
            "Injury",
            conf,
            probs,
            f"Structural cord break / deformity detected (deformity index {deform:.3f}). "
            "Cord contour is irregular or concave, consistent with traumatic spinal cord injury.",
        )

    # ------------------------------------------------------------------
    # RULE 2 - NORMAL: nothing found
    #   No bright lesions AND no cord deformity -> healthy cord.
    # ------------------------------------------------------------------
    if lesion_count == 0 or total_area_ratio < 0.005:
        probs = {"Tumor": 1.0, "MS": 1.5, "Injury": 1.5, "Normal": 96.0}
        return (
            "Normal",
            96.0,
            probs,
            "No abnormal bright regions and no structural deformity detected. "
            "Spinal cord parenchyma appears intact and homogeneous.",
        )

    # ------------------------------------------------------------------
    # RULE 3 - TUMOR: one large bright region
    #   A single dominant hyperintense focus that occupies >= 6% of the
    #   cord cross-section area.
    # ------------------------------------------------------------------
    is_large_lesion  = largest >= 0.06
    is_dominant_area = total_area_ratio >= 0.04 and lesion_count <= 2
    if is_large_lesion or is_dominant_area:
        raw_conf = min(97.0, 58.0 + largest * 450.0 + contrast * 120.0 + total_area_ratio * 180.0)
        conf = round(max(72.0, raw_conf), 2)
        rem  = round(100.0 - conf, 2)
        probs = {
            "Tumor":  conf,
            "MS":     round(rem * 0.15, 2),
            "Injury": round(rem * 0.15, 2),
            "Normal": round(rem * 0.70, 2),
        }
        return (
            "Tumor",
            conf,
            probs,
            f"Large focal hyperintense intramedullary lesion detected - occupies {largest * 100:.1f}% of "
            f"cord cross-section with {lesion_count} dominant focus(es). Consistent with spinal cord tumor.",
        )

    # ------------------------------------------------------------------
    # RULE 4 - MS: multiple small bright regions
    #   Several discrete small hyperintense plaques scattered along the
    #   cord, each individually < 6% of cord area.
    # ------------------------------------------------------------------
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
        )

    # ------------------------------------------------------------------
    # FALLBACK - score every class and pick the winner
    # ------------------------------------------------------------------
    normal_score = (
        (4.0 if total_area_ratio < 0.01 else -2.0)
        + (3.0 if lesion_count == 0 else -2.0)
        + (2.0 if deform < 0.04 else -1.5)
    )
    injury_score = (
        (5.0 if deform >= 0.05 else 0.0)
        + (2.5 if elong >= 2.0 else 0.0)
        + (1.5 if 0.01 <= total_area_ratio <= 0.20 else 0.0)
    )
    tumor_score = (
        (5.0 if largest >= 0.06 else -2.0)
        + (2.5 if total_area_ratio >= 0.04 else 0.0)
        + (1.5 if contrast >= 0.08 else 0.0)
        + (1.0 if lesion_count <= 2 else -1.0)
    )
    ms_score = (
        (4.0 if lesion_count >= 2 else 0.0)
        + (2.0 if 0.01 <= total_area_ratio <= 0.12 else -1.0)
        + (2.0 if largest < 0.06 else -3.0)
        + (1.0 if deform < 0.05 else -1.0)
    )

    scores    = np.array([tumor_score, ms_score, injury_score, normal_score], dtype=np.float64)
    probs_arr = _softmax(scores, temperature=1.1)

    idx        = int(np.argmax(probs_arr))
    label      = CLASS_NAMES[idx]
    confidence = round(float(probs_arr[idx]) * 100.0, 2)
    prob_dict  = {CLASS_NAMES[i]: round(float(probs_arr[i]) * 100.0, 2) for i in range(len(CLASS_NAMES))}

    reasoning_map = {
        "Normal":  f"No significant abnormality - anomaly area {total_area_ratio*100:.1f}% of cord.",
        "MS":      f"{lesion_count} small discrete plaques with preserved cord contour suggest MS.",
        "Tumor":   f"Dominant focal mass occupying {largest*100:.1f}% of cord favours a spinal tumor.",
        "Injury":  f"Cord deformity index {deform:.3f} and elongated morphology support traumatic injury.",
    }
    return label, confidence, prob_dict, reasoning_map.get(label, "")
'''

# Find the classify function block (from 'def classify(' to just before 'def make_overlay(')
pattern = re.compile(
    r'(def classify\(.*?\n)(.*?)((?=\ndef make_overlay))',
    re.DOTALL
)
m = pattern.search(src)
if not m:
    print('ERROR: classify() block not found')
    # Debug: show where classify starts
    idx = src.find('def classify(')
    print(f'classify found at index: {idx}')
    print(src[idx:idx+200])
else:
    new_src = src[:m.start()] + new_classify + '\n\n' + src[m.end():]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_src)
    print('Done. classify() replaced successfully.')
