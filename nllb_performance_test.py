import time
import sys
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 75)
print("LINGUASHIELD - NLLB PERFORMANCE TEST")
print("=" * 75)
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("Transformers:", transformers.__version__)
print("Device      :", DEVICE)
print("=" * 75)

print("\nLoading tokenizer and model...")
start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE).eval()

load_time = time.perf_counter() - start

tests = [
    ("Hindi", "hin_Deva", "अपना OTP अभी भेजें।"),
    ("Hindi", "hin_Deva", "अपना OTP किसी को साझा न करें।"),
    ("Hindi", "hin_Deva", "आपका बैंक खाता बंद कर दिया गया है। इसे तुरंत सत्यापित करें।"),
    ("Hindi", "hin_Deva", "इनाम पाने के लिए इस लिंक पर अभी क्लिक करें।"),
    ("Marathi", "mar_Deva", "तुमचा OTP आत्ताच पाठवा."),
    ("Marathi", "mar_Deva", "तुमचा OTP कोणासोबतही शेअर करू नका."),
    ("Marathi", "mar_Deva", "तुमचे बँक खाते बंद करण्यात आले आहे. त्वरित सत्यापित करा."),
    ("Marathi", "mar_Deva", "या लिंकवर लगेच क्लिक करा.")
]

# Warm-up
tokenizer.src_lang = tests[0][1]
warm_inputs = tokenizer(
    tests[0][2],
    return_tensors="pt",
    truncation=True
).to(DEVICE)

with torch.no_grad():
    model.generate(
        **warm_inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=128
    )

times = []

print("\nTranslations:")
for lang, src_lang, text in tests:
    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128
        )

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    result = tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )[0]

    print(f"{lang:8} | {elapsed:6.2f}s | {result}")

print("\n" + "=" * 75)
print(f"Model load time      : {load_time:.2f}s")
print(f"Average translation  : {sum(times)/len(times):.2f}s")
print(f"Fastest translation  : {min(times):.2f}s")
print(f"Slowest translation  : {max(times):.2f}s")
print(f"Total inference time : {sum(times):.2f}s")
print(f"Tests completed      : {len(times)}/{len(tests)}")
print("=" * 75)