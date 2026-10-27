import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cpu"

print("=" * 70)
print("LINGUASHIELD — NLLB TRANSLATION TEST")
print("=" * 70)

print(f"PyTorch: {torch.__version__}")
print(f"Device : {DEVICE}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

tests = [
    ("Hindi RISK", "अपना OTP अभी भेजें।", "hin_Deva"),
    ("Hindi SAFE", "अपना OTP किसी को साझा न करें।", "hin_Deva"),
    ("Marathi RISK", "तुमचा OTP आत्ताच पाठवा.", "mar_Deva"),
    ("Marathi SAFE", "तुमचा OTP कोणासोबतही शेअर करू नका.", "mar_Deva"),
]

for label, text, source_lang in tests:
    print("\n" + "-" * 70)
    print(label)
    print("Original:", text)

    tokenizer.src_lang = source_lang

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    target_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=target_lang_id,
            max_length=128,
            num_beams=5,
        )

    translated = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]

    print("English :", translated)

print("\n" + "=" * 70)
print("TRANSLATION TEST COMPLETE")
print("=" * 70)