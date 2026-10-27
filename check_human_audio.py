import os
import requests

FOLDER = r"C:\Users\ROHIT\Desktop\CODES\LinguaShield\backend\human_audio_samples"
API = "http://127.0.0.1:8000/api/analyze/file"

extensions = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".aiff", ".opus"}

files = [
    os.path.join(FOLDER, f)
    for f in os.listdir(FOLDER)
    if os.path.splitext(f)[1].lower() in extensions
]

print("=" * 80)
print("LINGUASHIELD - HUMAN AUDIO SAMPLE CHECK")
print("=" * 80)
print(f"FILES FOUND: {len(files)}")
print()

for path in sorted(files):
    name = os.path.basename(path)

    try:
        with open(path, "rb") as audio:
            r = requests.post(
                API,
                files={"file": (name, audio, "application/octet-stream")},
                data={"age_group": "adult"},
                timeout=180,
            )

        data = r.json()

        acoustic = data.get("acoustic_forensics", {})
        label = (
            acoustic.get("classification_label")
            or acoustic.get("decision_label")
            or data.get("classification_label")
            or "UNKNOWN"
        )
        synthetic = (
            acoustic.get("synthetic_detected")
            if "synthetic_detected" in acoustic
            else acoustic.get("is_synthetic_voice")
        )
        score = acoustic.get(
            "vocoder_score",
            acoustic.get("synthetic_probability")
        )
        confidence = acoustic.get(
            "confidence",
            acoustic.get("confidence_label")
        )

        print("-" * 80)
        print(f"FILE       : {name}")
        print(f"HTTP       : {r.status_code}")
        print(f"RESULT     : {label}")
        print(f"SYNTHETIC  : {synthetic}")
        print(f"SCORE      : {score}")
        print(f"CONFIDENCE : {confidence}")

    except Exception as e:
        print("-" * 80)
        print(f"FILE       : {name}")
        print(f"ERROR      : {e}")

print()
print("=" * 80)
print("CHECK COMPLETE")
print("=" * 80)
