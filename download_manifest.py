import sys
import os

# ─── DYNAMIC ENVIRONMENT ROUTER LAYER ───
current_dir = os.path.dirname(os.path.abspath(__file__))
local_site_packages = os.path.join(current_dir, ".venv", "Lib", "site-packages")

# Force path mapping to local project packages
if os.path.exists(local_site_packages):
    sys.path.insert(0, local_site_packages)

try:
    import pandas as pd
    from datasets import load_dataset
except ModuleNotFoundError as e:
    print(f"[!] Missing library: {e.name}. Run the pip install command first.")
    sys.exit(1)

print("[+] Connecting to Hugging Face Hub (Streaming Mode Active)...")

# Focusing on Hindi and Marathi code-mixed regional text corpora
target_languages = ["hindi", "marathi"]
compiled_rows = []
MAX_SAMPLES_PER_LANG = 5000  # Bounds limit to keep your local model lightweight and fast

for lang in target_languages:
    print(f"[➔] Opening live telemetry stream for dialect: {lang}...")
    try:
        # Load dataset with streaming=True
        dataset = load_dataset("ai4bharat/IndicVoices", lang, split="train", streaming=True)
        
        # 🛡️ THE COGNITIVE BYPASS FIX: 
        # Select ONLY the text field to drop the audio feature schema completely.
        # This prevents Hugging Face from requiring torchcodec / media decoders.
        try:
            dataset = dataset.select_columns(["text"])
        except Exception:
            try:
                dataset = dataset.remove_columns(["audio"])
            except Exception:
                pass

        count = 0
        for row in dataset:
            text_field = None
            for key in ["text", "transcription", "normalized_text", "sentence"]:
                if key in row and isinstance(row[key], str) and row[key].strip():
                    text_field = key
                    break
            
            if text_field:
                extracted_text = row[text_field].strip()
                compiled_rows.append({
                    "text": extracted_text,
                    "language_tag": lang
                })
                count += 1
                
            if count >= MAX_SAMPLES_PER_LANG:
                break
                
        print(f"[✓] Extracted {count} text manifest entries for {lang}.")
        
    except Exception as e:
        print(f"[!] Stream processing variance encountered for split '{lang}': {e}")

# Save the filtered results directly to your dataset environment layout
if compiled_rows:
    master_df = pd.DataFrame(compiled_rows)
    output_path = os.path.join(current_dir, "dataset", "indic_voices_text_manifest.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    master_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n==================================================")
    print(f"[✓] SUCCESS: Consolidated text manifest compiled.")
    print(f"    Target Location: {output_path}")
    print(f"    Total Training Rows Generated: {len(master_df)}")
    print(f"==================================================")
else:
    print("[!] Error: No data streams could be parsed. Check repository tokens.")