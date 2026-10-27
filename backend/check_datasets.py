import os
import pandas as pd

# Hardcoded directly to your local workspace destination path
dataset_dir = r"C:\Users\ROHIT\Desktop\Codes\LinguaShield\backend\dataset"
print("\n" + "="*60)
print(f"🔍 LINGUASHIELD DATASET HOUSING DIAGNOSTIC")
print("="*60)

if not os.path.exists(dataset_dir):
    print(f"[!] Target Directory Path Does Not Exist: {dataset_dir}")
else:
    # Scan directory contents
    all_files = os.listdir(dataset_dir)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    print(f"Total files detected in folder: {len(all_files)}")
    print(f"CSV files detected in folder:   {csv_files}\n")
    print("Checking internal schema layout rules...")
    print("-" * 60)
    
    for file_name in csv_files:
        if file_name == 'staging_feedback.csv':
            print(f"📄 File: {file_name} -> [Skipped automatically: System Live Logs]")
            print("-" * 60)
            continue
            
        file_path = os.path.join(dataset_dir, file_name)
        try:
            # Load only the first two index rows to examine structure instantly
            sample_df = pd.read_csv(file_path, nrows=2, encoding='utf-8', encoding_errors='replace')
            columns_list = list(sample_df.columns)
            
            has_label = 'label' in columns_list
            has_content = 'msg_content' in columns_list
            
            print(f"📄 File: {file_name}")
            print(f"   ↳ Columns Found:     {columns_list}")
            print(f"   ↳ Schema Validated?  {'[✓] PASS' if (has_label and has_content) else '[X] FAIL - Mismatched Headers'}")
        except Exception as read_fault:
            print(f"❌ File: {file_name} -> Critical Read Interruption: {read_fault}")
        print("-" * 60)
print("="*60 + "\n")