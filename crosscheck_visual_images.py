import os
import requests

FOLDER = r"C:\Users\ROHIT\Desktop\CODES\LinguaShield\backend\visual_test_images"
API = "http://127.0.0.1:8000/api/analyze/file"

extensions = {".png", ".jpg", ".jpeg", ".webp"}

files = [
    os.path.join(FOLDER, f)
    for f in os.listdir(FOLDER)
    if os.path.splitext(f)[1].lower() in extensions
]

print("=" * 90)
print("LINGUASHIELD - VISUAL TEST IMAGE CROSS-CHECK")
print("=" * 90)
print(f"IMAGES FOUND: {len(files)}")
print()

for path in sorted(files):
    name = os.path.basename(path)

    try:
        with open(path, "rb") as f:
            r = requests.post(
                API,
                files={"file": (name, f, "application/octet-stream")},
                data={"age_group": "adult"},
                timeout=180
            )

        data = r.json()

        print("-" * 90)
        print(f"FILE           : {name}")
        print(f"HTTP           : {r.status_code}")
        print(f"IS SCAM        : {data.get('is_scam')}")
        print(f"CLASSIFICATION : {data.get('risk_classification')}")
        print(f"CONFIDENCE     : {data.get('confidence')}")
        print(f"EXTRACTED TEXT : {data.get('extracted_text', '')[:250]}")

    except Exception as e:
        print("-" * 90)
        print(f"FILE  : {name}")
        print(f"ERROR : {e}")

print()
print("=" * 90)
print("VISUAL CROSS-CHECK COMPLETE")
print("=" * 90)
