import sys
from pathlib import Path
from starlette.testclient import TestClient

sys.path.append(str(Path(__file__).parent / "backend"))
sys.path.append(str(Path(__file__).parent / "ml"))

from server import app, CLASS_NAMES, ML_MODEL, ML_MODELS

print(f"CLASS_NAMES: {CLASS_NAMES}", flush=True)
print(f"ML_MODEL: {type(ML_MODEL).__name__ if ML_MODEL else None}", flush=True)
print(f"ML_MODELS keys: {list(ML_MODELS.keys())}", flush=True)

client = TestClient(app)
data_dir = Path(__file__).parent / "ml" / "data"

for c in ["Tumor", "MS", "Injury", "Normal"]:
    img_path = data_dir / c / "images" / f"{c}_000.png"
    if not img_path.exists():
        print(f"Image not found: {img_path}", flush=True)
        continue
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    
    response = client.post(
        "/api/predict",
        files={"file": (img_path.name, file_bytes, "image/png")},
        data={"actual_class": c, "model_type": "neurocord_resattn"}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"\n[EVALUATION] Ground Truth: {c:<8} -> Predicted: {data['predicted_class']:<8} (Confidence: {data['confidence']:.1f}%)", flush=True)
        print(f"  Reasoning: {data['reasoning']}", flush=True)
        print(f"  Dice Score: {data['dice_score']:.3f} | Lesions: {data['lesion_count']} | Area Ratio: {data['lesion_area_ratio']:.4f}", flush=True)
        print(f"  Probabilities: {data['probabilities']}", flush=True)
        print(f"  Grad-CAM Generated: {'Yes' if data.get('gradcam_b64') else 'No'}", flush=True)
        print(f"  SHAP Generated: {'Yes' if data.get('shap_b64') else 'No'}", flush=True)
    else:
        print(f"Error {response.status_code}: {response.text}", flush=True)
