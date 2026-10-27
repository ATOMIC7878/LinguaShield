import pandas as pd

# The path to your 3rd massive dataset file
FILE_PATH = "backend/dataset/massive_phishing.csv"

print("==================================================")
print("     LINGUASHIELD - DATASET DEEP PROFILER        ")
print("==================================================\n")

try:
    # Read the data using Latin-1 encoding to prevent character decoding errors
    df = pd.read_csv(FILE_PATH, encoding='latin-1')
    
    # Ensure columns exist
    if 'text_combined' not in df.columns or 'label' not in df.columns:
        print("[!] Critical Error: Could not find 'text_combined' or 'label' in this file.")
        print(f"    Available columns are: {df.columns.tolist()}")
        exit()

    # 1. Row Counts & Class Balance
    total_rows = len(df)
    class_counts = df['label'].value_counts()
    
    print("--- 1. SET COMPOSITION & CLASS BALANCE ---")
    print(f" * Total Dataset Volume: {total_rows:,} rows")
    for label, count in class_counts.items():
        percentage = (count / total_rows) * 100
        identity = "Scam / Phishing" if int(label) == 1 else "Safe / Legitimate"
        print(f" * {identity} ({label}): {count:,} rows ({percentage:.2f}%)")
        
    # 2. Text Density Profiling (SMS vs. Long Emails)
    # Convert column to string to safely process any missing entries
    df['char_length'] = df['text_combined'].astype(str).apply(len)
    avg_length = df['char_length'].mean()
    max_length = df['char_length'].max()
    
    print("\n--- 2. LINGUISTIC TEXT DENSITY PROFILE ---")
    print(f" * Average Message Length: {avg_length:.1f} characters")
    print(f" * Longest Message Length: {max_length:,} characters")
    
    if avg_length < 200:
        print(" * Profile Deduction: This set contains mostly short conversational alerts (SMS style).")
    else:
        print(" * Profile Deduction: This set contains long-form text patterns (Email/Document style).")

    # 3. Direct Psychological Threat Extractions
    print("\n--- 3. RAW SCAM TARGET SAMPLES (Label = 1) ---")
    scam_subset = df[df['label'] == 1].head(2)
    
    if len(scam_subset) == 0:
        print(" * No rows found with a scam label of 1.")
    else:
        for idx, row in enumerate(scam_subset.itertuples(), 1):
            raw_text = str(row.text_combined).strip()
            # Clip the text preview clean if it's an incredibly massive email block
            preview = raw_text[:350] + " ... [Truncated]" if len(raw_text) > 350 else raw_text
            print(f"\n [Threat Sample #{idx}]:")
            print(f" \"{preview}\"")
            print("-" * 50)

except FileNotFoundError:
    print(f"[!] Error: System could not find the file at: {FILE_PATH}")
    print("    Make sure your large file is moved and renamed inside backend/dataset/")
except Exception as e:
    print(f"[!] An unexpected error occurred: {e}")