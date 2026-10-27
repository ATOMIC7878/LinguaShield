# LinguaShield Text Pipeline Test Runner
# Run from the project ROOT:
#   PS C:\Users\ROHIT\Desktop\CODES\LinguaShield> python test_text_pipeline.py
#
# This script tests the shared /api/analyze/text pipeline with 20 cases:
# 10 attack cases (expected True) and 10 awareness/safe cases (expected False).
#
# It prints every API response summary and a final PASS/FAIL table.
# It does NOT modify your backend or frontend.

import json
import sys
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/api/analyze/text"

TESTS = [
    ("1_ACCOUNT_SECURITY_ATTACK", True,
     "URGENT: We detected unusual activity on your account. Send your OTP and password immediately to avoid suspension. Your account will be suspended in 10 minutes."),

    ("2_ACCOUNT_SECURITY_AWARENESS", False,
     "ACCOUNT SECURITY AWARENESS: A legitimate bank or service provider will never ask for your OTP or password through a phone call, email, or message. Do not share your confidential information with anyone. If you receive a suspicious message, ignore it and independently check your account through the official website or customer service."),

    ("3_DIGITAL_ARREST_ATTACK", True,
     "URGENT: You are involved in a serious criminal case and a digital arrest order has been issued. Immediately transfer the required money to complete the payment. Do not discuss this with anyone."),

    ("4_DIGITAL_ARREST_AWARENESS", False,
     "DIGITAL ARREST SECURITY WARNING: This is a fake scam. Real police officers do not digitally arrest people or ask them to transfer money. End the conversation and independently verify the situation through official police or government channels."),

    ("5_BANK_SECURITY_ATTACK", True,
     "URGENT BANK SECURITY NOTICE: The bank security department detected suspicious activity on your account. Banking services will be disabled today. Send your OTP, PIN and CVV immediately within 15 minutes to restore access."),

    ("6_BANK_SECURITY_AWARENESS", False,
     "BANK SECURITY AWARENESS: A legitimate bank will never ask for your OTP, PIN or CVV. If someone claims to be a bank employee, do not share confidential information. Contact your bank independently using official contact details."),

    ("7_REMOTE_ACCESS_ATTACK", True,
     "URGENT TECHNICAL SECURITY NOTICE: There is a security problem with your computer. Download and install AnyDesk, allow screen sharing and send the access code so we can secure your computer."),

    ("8_REMOTE_ACCESS_AWARENESS", False,
     "REMOTE ACCESS SECURITY AWARENESS: AnyDesk and TeamViewer can be legitimate support tools, but do not install them at the request of an unknown person and never give them your access code. Contact your official IT support instead."),

    ("9_MICROSOFT_ATTACK", True,
     "MICROSOFT 365 SECURITY ALERT: We detected a suspicious sign-in. Verify your identity using https://microsoft-account-security.example.com/verify before the deadline or your account will be suspended."),

    ("10_PHISHING_AWARENESS", False,
     "PHISHING SECURITY AWARENESS: Phishing messages may pretend to be Microsoft, banks or other trusted organizations. Do not click unexpected links. Open the official website yourself and report suspicious messages."),

    ("11_CEO_PAYMENT_ATTACK", True,
     "CONFIDENTIAL CEO PAYMENT REQUEST: I am the CEO. We have a confidential acquisition matter. Wire transfer ₹12,50,000 to the external legal advisor within 30 minutes. Do not discuss this with anyone and send confirmation."),

    ("12_CEO_AWARENESS", False,
     "CEO IMPERSONATION SECURITY AWARENESS: Scams may ask for urgent payments or try to bypass normal approval procedures. Independently verify requests through the official directory or a trusted contact. Never share confidential financial information or payment confirmations."),

    ("13_CRYPTO_ATTACK", True,
     "GUARANTEED CRYPTO INVESTMENT OPPORTUNITY: Invest ₹25,000 and double your money in 24 hours. Transfer crypto to this wallet. The opportunity expires today. Complete the payment within 30 minutes. Guaranteed profit."),

    ("14_CRYPTO_AWARENESS", False,
     "CRYPTO INVESTMENT SECURITY AWARENESS: Crypto and stock investments involve risk. No legitimate investment plan can guarantee doubling your money or a fixed profit. Independently verify investment opportunities and remain cautious."),

    ("15_CRYPTO_ATTACK_DUPLICATE", True,
     "GUARANTEED CRYPTO INVESTMENT OPPORTUNITY: Invest ₹25,000 and double your money in 24 hours. Transfer crypto to this wallet. The opportunity expires today. Complete the payment within 30 minutes. Guaranteed profit."),

    ("16_CRYPTO_AWARENESS_DUPLICATE", False,
     "CRYPTO INVESTMENT SECURITY AWARENESS: Crypto and stock investments involve risk. No legitimate investment plan can guarantee doubling your money or a fixed profit. Independently verify investment opportunities and remain cautious."),

    ("17_REFUND_ATTACK", True,
     "URGENT REFUND NOTICE: An additional ₹18,500 is pending. To receive your refund, pay a ₹1,250 processing fee. Transfer the fee immediately and complete the payment within 20 minutes."),

    ("18_REFUND_AWARENESS", False,
     "REFUND SECURITY AWARENESS: Do not pay a processing fee or make an additional payment to receive a refund. Independently verify refund or invoice information through official channels."),

    ("19_ROMANCE_ATTACK", True,
     "PRIVATE AND URGENT HELP: After weeks of trust, I am stranded abroad with a medical emergency. I urgently need ₹75,000 for the hospital bill. Please transfer the money to my bank account. I will repay you. Do not tell your family or friends."),

    ("20_ROMANCE_AWARENESS", False,
     "ROMANCE SCAM SECURITY AWARENESS: Online relationships can sometimes lead to requests for money. Independently verify the person's identity. Do not send money because of an emergency, secrecy or emotional pressure. Speak with someone you trust."),
]


def call_api(text: str):
    payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("=" * 88)
    print("LINGUASHIELD — SHARED TEXT PIPELINE FALSE-ALARM TEST")
    print("=" * 88)
    print(f"API: {API_URL}")
    print(f"Cases: {len(TESTS)}")
    print()

    results = []
    passed = 0

    for index, (name, expected, text) in enumerate(TESTS, start=1):
        print(f"[{index:02d}/{len(TESTS)}] {name}")

        try:
            data = call_api(text)

            actual = bool(data.get("is_scam"))
            confidence = data.get("confidence", "-")
            risk = data.get("risk_classification", "-")

            context = data.get("text_context") or {}
            context_class = context.get("context_class", "-")
            decision_basis = context.get("decision_basis", "-")
            defensive = context.get("defensive_context", "-")

            ok = actual == expected
            if ok:
                passed += 1

            status = "PASS" if ok else "FAIL"
            results.append((name, expected, actual, confidence, risk, context_class, status))

            print(f"      Expected: {'THREAT' if expected else 'CLEAR'}")
            print(f"      Actual:   {'THREAT' if actual else 'CLEAR'}")
            print(f"      Status:   {status}")
            print(f"      Confidence: {confidence}")
            print(f"      Risk: {risk}")
            print(f"      Context: {context_class}")
            print(f"      Basis: {decision_basis}")
            print(f"      Defensive context: {defensive}")
            print()

        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            results.append((name, expected, "ERROR", "-", body[:120], "-", "ERROR"))
            print(f"      ERROR: HTTP {exc.code}")
            if body:
                print(f"      Response: {body[:500]}")
            print()

        except Exception as exc:
            results.append((name, expected, "ERROR", "-", str(exc), "-", "ERROR"))
            print(f"      ERROR: {exc}")
            print()

    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(f"Passed: {passed}/{len(TESTS)}")
    print(f"Failed: {len(TESTS) - passed}/{len(TESTS)}")
    print()

    print(f"{'CASE':<38} {'EXP':<7} {'ACTUAL':<8} {'CONF':<8} {'STATUS'}")
    print("-" * 88)

    for name, expected, actual, confidence, risk, context_class, status in results:
        exp_text = "THREAT" if expected else "CLEAR"
        if actual == "ERROR":
            act_text = "ERROR"
        else:
            act_text = "THREAT" if actual else "CLEAR"
        print(f"{name:<38} {exp_text:<7} {act_text:<8} {str(confidence):<8} {status}")

    print()
    if passed == len(TESTS):
        print("RESULT: ALL TESTS PASSED.")
        print("The shared Text analysis pipeline produced the expected verdict for every case.")
        print("For Visual, the remaining separate risk is OCR/image preprocessing, not this shared text decision path.")
        return 0

    print("RESULT: FAILURES DETECTED.")
    print("At least one case does not match the expected verdict.")
    print("Review the per-case Context/Basis/Risk above before changing any code.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
