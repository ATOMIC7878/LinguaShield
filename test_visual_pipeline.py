# LinguaShield Visual Pipeline Test Runner
#
# Run this from the LinguaShield PROJECT ROOT:
#
#   PS C:\Users\ROHIT\Desktop\CODES\LinguaShield> python test_visual_pipeline.py
#
# Test images:
#   C:\Users\ROHIT\Desktop\CODES\LinguaShield\backend\visual_test_images
#
# Expected image numbering (18 unique cases):
#   1  Account Security Attack
#   2  Account Security Awareness
#   3  Digital Arrest Attack
#   4  Digital Arrest Awareness
#   5  Bank Security Attack
#   6  Bank Security Awareness
#   7  Remote Access Attack
#   8  Remote Access Awareness
#   9  Microsoft 365 Attack
#   10 Phishing Awareness
#   11 CEO Payment Attack
#   12 CEO Impersonation Awareness
#   13 Crypto Investment Attack
#   14 Crypto Investment Awareness
#   15 Refund Attack
#   16 Refund Awareness
#   17 Romance Attack
#   18 Romance Awareness
#
# Note:
# The duplicate crypto cases from the 20-text test are intentionally represented
# only once here because you said you have 18 image files.
#
# This script ONLY calls the running API. It does not modify your backend/frontend.

import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


API_URL = "http://127.0.0.1:8000/api/analyze/file"
IMAGE_DIR = Path(r"C:\Users\ROHIT\Desktop\CODES\LinguaShield\backend\visual_test_images")

TESTS = [
    (1, "ACCOUNT_SECURITY_ATTACK", True),
    (2, "ACCOUNT_SECURITY_AWARENESS", False),
    (3, "DIGITAL_ARREST_ATTACK", True),
    (4, "DIGITAL_ARREST_AWARENESS", False),
    (5, "BANK_SECURITY_ATTACK", True),
    (6, "BANK_SECURITY_AWARENESS", False),
    (7, "REMOTE_ACCESS_ATTACK", True),
    (8, "REMOTE_ACCESS_AWARENESS", False),
    (9, "MICROSOFT_365_ATTACK", True),
    (10, "PHISHING_AWARENESS", False),
    (11, "CEO_PAYMENT_ATTACK", True),
    (12, "CEO_IMPERSONATION_AWARENESS", False),
    (13, "CRYPTO_INVESTMENT_ATTACK", True),
    (14, "CRYPTO_INVESTMENT_AWARENESS", False),
    (15, "REFUND_ATTACK", True),
    (16, "REFUND_AWARENESS", False),
    (17, "ROMANCE_ATTACK", True),
    (18, "ROMANCE_AWARENESS", False),
]


def multipart_body(file_path: Path):
    boundary = "----LinguaShieldBoundary" + uuid.uuid4().hex
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    file_bytes = file_path.read_bytes()

    chunks = []
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n"
            f"\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(
        f'\r\n--{boundary}\r\n'
        f'Content-Disposition: form-data; name="age_group"\r\n'
        f"\r\n"
        f"adult\r\n"
        f"--{boundary}--\r\n".encode("utf-8")
    )

    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def call_visual_api(file_path: Path):
    body, content_type = multipart_body(file_path)

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": content_type,
            "Accept": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def compact_text(value, max_len=72):
    if value is None:
        return "-"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main():
    print("=" * 105)
    print("LINGUASHIELD — VISUAL / OCR / SHARED PIPELINE TEST")
    print("=" * 105)
    print(f"API:       {API_URL}")
    print(f"IMAGE DIR: {IMAGE_DIR}")
    print()

    if not IMAGE_DIR.exists():
        print("ERROR: Image folder does not exist:")
        print(IMAGE_DIR)
        print()
        print(r'Create/use: C:\Users\ROHIT\Desktop\CODES\LinguaShield\backend\visual_test_images')
        return 1

    missing = []
    files = {}

    for number, _, _ in TESTS:
        candidates = [
            IMAGE_DIR / f"{number}.png",
            IMAGE_DIR / f"{number}.jpg",
            IMAGE_DIR / f"{number}.jpeg",
            IMAGE_DIR / f"{number}.webp",
        ]

        found = next((p for p in candidates if p.exists()), None)

        if found is None:
            missing.append(number)
        else:
            files[number] = found

    if missing:
        print(f"ERROR: Missing image file(s): {', '.join(map(str, missing))}")
        print()
        print("Expected names:")
        print("1.png ... 18.png (or .jpg/.jpeg/.webp)")
        return 1

    print(f"Found all {len(TESTS)} image files.")
    print()

    results = []

    for number, name, expected in TESTS:
        file_path = files[number]
        print(f"[{number:02d}/18] {name}  ->  {file_path.name}")

        try:
            data = call_visual_api(file_path)

            actual = bool(data.get("is_scam"))
            confidence = data.get("confidence", "-")
            risk = data.get("risk_classification", "-")
            processed_text = data.get("processed_text", "")
            context = data.get("text_context") or {}

            context_class = context.get("context_class", "-")
            defensive = context.get("defensive_context", "-")
            directive = context.get("malicious_directive_detected", "-")
            basis = context.get("decision_basis", "-")
            model_prediction = context.get("model_prediction", "-")
            clear_applied = context.get("context_clear_applied", "-")

            passed = actual == expected
            status = "PASS" if passed else "FAIL"

            results.append(
                {
                    "number": number,
                    "name": name,
                    "expected": expected,
                    "actual": actual,
                    "confidence": confidence,
                    "risk": risk,
                    "context_class": context_class,
                    "defensive": defensive,
                    "directive": directive,
                    "basis": basis,
                    "model_prediction": model_prediction,
                    "clear_applied": clear_applied,
                    "processed_text": processed_text,
                    "status": status,
                }
            )

            print(f"      Expected : {'THREAT' if expected else 'CLEAR'}")
            print(f"      Actual   : {'THREAT' if actual else 'CLEAR'}")
            print(f"      Status   : {status}")
            print(f"      Confidence: {confidence}")
            print(f"      Risk     : {compact_text(risk, 95)}")
            print(f"      Context  : {context_class}")
            print(f"      Defensive: {defensive}")
            print(f"      Directive: {directive}")
            print(f"      Basis    : {basis}")
            print(f"      OCR text : {compact_text(processed_text, 110)}")
            print()

        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            results.append(
                {
                    "number": number,
                    "name": name,
                    "expected": expected,
                    "actual": "ERROR",
                    "confidence": "-",
                    "risk": f"HTTP {exc.code}: {body[:200]}",
                    "context_class": "-",
                    "defensive": "-",
                    "directive": "-",
                    "basis": "-",
                    "model_prediction": "-",
                    "clear_applied": "-",
                    "processed_text": "",
                    "status": "ERROR",
                }
            )

            print(f"      ERROR: HTTP {exc.code}")
            if body:
                print(f"      Response: {body[:500]}")
            print()

        except Exception as exc:
            results.append(
                {
                    "number": number,
                    "name": name,
                    "expected": expected,
                    "actual": "ERROR",
                    "confidence": "-",
                    "risk": str(exc),
                    "context_class": "-",
                    "defensive": "-",
                    "directive": "-",
                    "basis": "-",
                    "model_prediction": "-",
                    "clear_applied": "-",
                    "processed_text": "",
                    "status": "ERROR",
                }
            )

            print(f"      ERROR: {exc}")
            print()

    passed_count = sum(r["status"] == "PASS" for r in results)
    failed_count = sum(r["status"] == "FAIL" for r in results)
    error_count = sum(r["status"] == "ERROR" for r in results)

    print("=" * 105)
    print("FINAL SUMMARY")
    print("=" * 105)
    print(f"Passed : {passed_count}/{len(TESTS)}")
    print(f"Failed : {failed_count}/{len(TESTS)}")
    print(f"Errors : {error_count}/{len(TESTS)}")
    print()

    print(f"{'#':<4} {'CASE':<34} {'EXP':<8} {'ACTUAL':<8} {'CONF':<8} {'STATUS'}")
    print("-" * 105)

    for r in results:
        expected_text = "THREAT" if r["expected"] else "CLEAR"
        actual = r["actual"]

        if actual == "ERROR":
            actual_text = "ERROR"
        else:
            actual_text = "THREAT" if actual else "CLEAR"

        print(
            f'{r["number"]:<4} '
            f'{r["name"]:<34} '
            f'{expected_text:<8} '
            f'{actual_text:<8} '
            f'{str(r["confidence"]):<8} '
            f'{r["status"]}'
        )

    print()
    print("=" * 105)

    if passed_count == len(TESTS):
        print("RESULT: ALL 18 VISUAL TESTS PASSED.")
        print()
        print("The tested images successfully produced the expected verdicts through:")
        print("IMAGE -> OCR -> text normalization -> shared detection -> Visual response")
        print()
        print("This does not prove every possible image will be correct, but it confirms")
        print("this 18-image regression set passes end-to-end.")
        return 0

    print("RESULT: VISUAL TEST FAILURES DETECTED.")
    print()
    print("Important:")
    print("- FAIL on an expected THREAT = missed threat / false negative.")
    print("- FAIL on an expected CLEAR = false alarm / false positive.")
    print("- ERROR = API, file, OCR, or server error.")
    print()
    print("Use the per-case OCR text and decision basis above to determine whether")
    print("a failure comes from OCR/image extraction or the shared text decision logic.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
