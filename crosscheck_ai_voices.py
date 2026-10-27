import os
import requests

FOLDER = r"C:\Users\ROHIT\Desktop\CODES\LinguaShield\backend\test_audio_samples"
API = "http://127.0.0.1:8000/api/analyze/file"

extensions = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".aiff", ".opus"}

files = [
    os.path.join(FOLDER, f)
    for f in os.listdir(FOLDER)
    if os.path.splitext(f)[1].lower() in extensions
]

print("=" * 90)
print("LINGUASHIELD - AI VOICE CROSS-CHECK")
print("=" * 90)
print(f"FILES FOUND: {len(files)}")
print()

passed = 0

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
        acoustic = data.get("acoustic_forensics", {})

        synthetic = acoustic.get("synthetic_detected")
        if synthetic is None:
            synthetic = acoustic.get("is_synthetic_voice")

        score = acoustic.get("vocoder_score")
        if score is None:
            score = acoustic.get("synthetic_probability")

        label = acoustic.get("decision_label", "UNKNOWN")
        confidence = acoustic.get("confidence_label", "UNKNOWN")

        ok = synthetic is True

        if ok:
            passed += 1

        print("-" * 90)
        print(f"FILE       : {name}")
        print(f"HTTP       : {r.status_code}")
        print(f"RESULT     : {label}")
        print(f"SYNTHETIC  : {synthetic}")
        print(f"SCORE      : {score}")
        print(f"CONFIDENCE : {confidence}")
        print(f"CROSS-CHECK: {'PASS' if ok else 'FAIL'}")

    except Exception as e:
        print("-" * 90)
        print(f"FILE       : {name}")
        print(f"ERROR      : {e}")

print()
print("=" * 90)
print(f"RESULT: {passed}/{len(files)} AI VOICE CROSS-CHECK PASS")
print("=" * 90)
