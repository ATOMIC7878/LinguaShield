import json
import requests

URL = "http://127.0.0.1:8000/api/analyze/text"

tests = [
    ("Hindi RISK", "\u0905\u092a\u0928\u093e OTP \u0905\u092d\u0940 \u092d\u0947\u091c\u0947\u0902\u0964"),
    ("Hindi SAFE", "\u0905\u092a\u0928\u093e OTP \u0915\u093f\u0938\u0940 \u0915\u094b \u0938\u093e\u091d\u093e \u0928 \u0915\u0930\u0947\u0902\u0964"),
    ("Hindi BANK RISK", "\u0906\u092a\u0915\u093e \u092c\u0948\u0902\u0915 \u0916\u093e\u0924\u093e \u092c\u0902\u0926 \u0915\u0930 \u0926\u093f\u092f\u093e \u0917\u092f\u093e \u0939\u0948\u0964 \u0907\u0938\u0947 \u0924\u0941\u0930\u0902\u0924 \u0938\u0924\u094d\u092f\u093e\u092a\u093f\u0924 \u0915\u0930\u0947\u0902\u0964"),
    ("Marathi RISK", "\u0924\u0941\u092e\u091a\u093e OTP \u0906\u0924\u094d\u0924\u093e\u091a \u092a\u093e\u0920\u0935\u093e."),
    ("Marathi SAFE", "\u0924\u0941\u092e\u091a\u093e OTP \u0915\u094b\u0923\u093e\u0938\u094b\u092c\u0924\u0939\u0940 \u0936\u0947\u0905\u0930 \u0915\u0930\u0942 \u0928\u0915\u093e."),
    ("Marathi BANK RISK", "\u0924\u0941\u092e\u091a\u0947 \u092c\u0901\u0915 \u0916\u093e\u0924\u0947 \u092c\u0902\u0926 \u0915\u0930\u0923\u094d\u092f\u093e\u0924 \u0906\u0932\u0947 \u0906\u0939\u0947. \u0924\u094d\u0935\u0930\u093f\u0924 \u0938\u0924\u094d\u092f\u093e\u092a\u093f\u0924 \u0915\u0930\u093e."),
    ("English RISK", "Send your OTP now."),
    ("English SAFE", "Do not share your OTP with anyone.")
]

print("=" * 75)
print("LINGUASHIELD - REAL API TEXT + NLLB TEST")
print("=" * 75)

for label, text in tests:
    print("\n" + "-" * 75)
    print(label)
    print("INPUT:", text)

    try:
        response = requests.post(
            URL,
            json={"text": text, "age_group": "adult"},
            timeout=120
        )

        print("HTTP STATUS:", response.status_code)

        try:
            data = response.json()
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            print("RAW RESPONSE:", response.text)

    except Exception as exc:
        print("REQUEST ERROR:", type(exc).__name__, str(exc))

print("\n" + "=" * 75)
print("REAL API TEXT + NLLB TEST COMPLETE")
print("=" * 75)