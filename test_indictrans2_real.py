import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("LINGUASHIELD — REAL INDIC TRANS 2 TEST")
print("=" * 70)
print("PyTorch:", torch.__version__)
print("Device :", DEVICE)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.to(DEVICE)
model.eval()

print("\n" + "-" * 70)

tests = [
    "Send your OTP now.",
    "Do not share your OTP with anyone.",
    "Your bank account has been blocked. Verify it immediately.",
    "Please click the link to claim your reward.",
]

for text in tests:
    print("English :", text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=256
        )

    output = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )[0]

    print("Output  :", output)
    print("-" * 70)

print("\nINDIC TRANS 2 TEST COMPLETE")