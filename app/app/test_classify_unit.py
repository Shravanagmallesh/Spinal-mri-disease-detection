import numpy as np

CLASS_NAMES = ['Tumor', 'MS', 'Injury', 'Normal']

def classify(lesion_count, total_area_ratio, largest, mean_comp, contrast, deform, elong, std_i):
    if deform >= 0.05 or (elong >= 2.5 and total_area_ratio >= 0.010):
        raw_conf = min(97.0, 68.0 + deform * 280.0 + elong * 3.0)
        conf = round(max(75.0, raw_conf), 2)
        return 'Injury', conf
    if lesion_count == 0 or total_area_ratio < 0.005:
        return 'Normal', 96.0
    is_large = largest >= 0.06
    is_dom   = total_area_ratio >= 0.04 and lesion_count <= 2
    if is_large or is_dom:
        raw_conf = min(97.0, 58.0 + largest*450 + contrast*120 + total_area_ratio*180)
        return 'Tumor', round(max(72.0, raw_conf), 2)
    if lesion_count >= 2 and largest < 0.06:
        raw_conf = min(95.0, 52.0 + lesion_count*9 + total_area_ratio*220)
        return 'MS', round(max(68.0, raw_conf), 2)
    return 'Fallback', 0.0

tests = [
    # (expected, lesion_count, total_area_ratio, largest, mean_comp, contrast, deform, elong, std_i)
    ('Tumor',  1, 0.12,  0.10,  0.05, 0.15, 0.01, 1.2, 0.10),   # one large bright region
    ('MS',     4, 0.04,  0.025, 0.02, 0.05, 0.01, 1.1, 0.06),   # 4 small bright regions
    ('Injury', 0, 0.002, 0.0,   0.0,  0.0,  0.10, 1.0, 0.02),   # cord break, zero bright lesions
    ('Normal', 0, 0.001, 0.0,   0.0,  0.0,  0.01, 1.0, 0.01),   # nothing found
    ('Injury', 1, 0.020, 0.015, 0.01, 0.02, 0.03, 3.0, 0.03),   # deformed + elongated
]

all_ok = True
for row in tests:
    expected = row[0]
    args = row[1:]
    label, conf = classify(*args)
    ok = label == expected
    all_ok = all_ok and ok
    status = 'PASS' if ok else 'FAIL'
    print(status + '  Expected=' + expected + '  Got=' + label + '  (' + str(conf) + '%)')

print()
print('All tests passed!' if all_ok else 'SOME TESTS FAILED')
