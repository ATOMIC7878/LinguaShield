import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 75)
print("LINGUASHIELD - NLLB COMPREHENSIVE TRANSLATION TEST")
print("=" * 75)
print("Python      :", __import__("sys").version.split()[0])
print("PyTorch     :", torch.__version__)
print("Transformers:", transformers.__version__)
print("Device      :", DEVICE)
print("=" * 75)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()

tests = [
    ("Hindi RISK",   "hin_Deva", "\u0905\u092a\u0928\u093e OTP \u0905\u092d\u0940 \u092d\u0947\u091c\u0947\u0902\u0964"),
    ("Hindi SAFE",   "hin_Deva", "\u0905\u092a\u0928\u093e OTP \u0915\u093f\u0938\u0940 \u0915\u094b \u0938\u093e\u091d\u093e \u0928 \u0915\u0930\u0947\u0902\u0964"),
    ("Hindi RISK",   "hin_Deva", "\u0906\u092a\u0915\u093e \u092c\u0948\u0902\u0915 \u0916\u093e\u0924\u093e \u092c\u0902\u0926 \u0915\u0930 \u0926\u093f\u092f\u093e \u0917\u092f\u093e \u0939\u0948\u0964 \u0907\u0938\u0947 \u0924\u0941\u0930\u0902\u0924 \u0938\u0924\u094d\u092f\u093e\u092a\u093f\u0924 \u0915\u0930\u0947\u0902\u0964"),
    ("Hindi RISK",   "hin_Deva", "\u0907\u0928\u093e\u092e \u092a\u093e\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0907\u0938 \u0932\u093f\u0902\u0915 \u092a\u0930 \u0905\u092d\u0940 \u0915\u094d\u0932\u093f\u0915 \u0915\u0930\u0947\u0902\u0964"),
    ("Hindi RISK",   "hin_Deva", "\u0930\u093f\u092e\u094b\u091f \u090f\u0915\u094d\u0938\u0947\u0938 \u0915\u0947 \u0932\u093f\u090f AnyDesk \u0907\u0902\u0938\u094d\u091f\u0949\u0932 \u0915\u0930\u0947\u0902\u0964"),
    ("Hindi SAFE",   "hin_Deva", "\u092c\u0948\u0902\u0915 \u0915\u092d\u0940 \u092d\u0940 \u092b\u094b\u0928 \u092a\u0930 \u0906\u092a\u0915\u093e OTP \u0928\u0939\u0940\u0902 \u092e\u093e\u0902\u0917\u0924\u093e\u0964"),

    ("Marathi RISK", "mar_Deva", "\u0924\u0941\u092e\u091a\u093e OTP \u0906\u0924\u094d\u0924\u093e\u091a \u092a\u093e\u0920\u0935\u093e."),
    ("Marathi SAFE", "mar_Deva", "\u0924\u0941\u092e\u091a\u093e OTP \u0915\u094b\u0923\u093e\u0938\u094b\u092c\u0924\u0939\u0940 \u0936\u0947\u0905\u0930 \u0915\u0930\u0942 \u0928\u0915\u093e."),
    ("Marathi RISK", "mar_Deva", "\u0924\u0941\u092e\u091a\u0947 \u092c\u0901\u0915 \u0916\u093e\u0924\u0947 \u092c\u0902\u0926 \u0915\u0930\u0923\u094d\u092f\u093e\u0924 \u0906\u0932\u0947 \u0906\u0939\u0947. \u0924\u094d\u0935\u0930\u093f\u0924 \u0938\u0924\u094d\u092f\u093e\u092a\u093f\u0924 \u0915\u0930\u093e."),
    ("Marathi RISK", "mar_Deva", "\u092c\u0915\u094d\u0937\u0940\u0938 \u092e\u093f\u0933\u0935\u0923\u094d\u092f\u093e\u0938\u093e\u0920\u0940 \u092f\u093e \u0932\u093f\u0902\u0915\u0935\u0930 \u0932\u0917\u0947\u091a \u0915\u094d\u0932\u093f\u0915 \u0915\u0930\u093e."),
    ("Marathi RISK", "mar_Deva", "\u0930\u093f\u092e\u094b\u091f \u0905\u0945\u0915\u094d\u0938\u0947\u0938\u0938\u093e\u0920\u0940 AnyDesk \u0907\u0928\u094d\u0938\u094d\u091f\u0949\u0932 \u0915\u0930\u093e."),
    ("Marathi SAFE", "mar_Deva", "\u092c\u0901\u0915 \u092b\u094b\u0928\u0935\u0930 \u0924\u0941\u092e\u091a\u093e OTP \u0915\u0927\u0940\u0939\u0940 \u092e\u093e\u0917\u0924 \u0928\u093e\u0939\u0940.")
]

success = 0

for label, src_lang, text in tests:
    print("\n" + "-" * 75)
    print(label)
    print("Original:", text)

    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=256
        )

    translation = tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True
    )[0]

    print("English :", translation)

    if translation.strip():
        success += 1

print("\n" + "=" * 75)
print(f"RESULT: {success}/{len(tests)} translations produced output")
print("=" * 75)
print("NLLB COMPREHENSIVE TEST COMPLETE")