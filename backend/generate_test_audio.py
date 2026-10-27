import sys
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
local_site_packages = os.path.join(project_root, ".venv", "Lib", "site-packages")

if os.path.exists(local_site_packages):
    sys.path.insert(0, local_site_packages)

try:
    import asyncio
    import edge_tts
except ModuleNotFoundError as e:
    print(f"Missing Module: '{e.name}'. Run: pip install edge-tts")
    sys.exit(1)

# 🌟 Expanded 5 Real-Life Scam Testing Profiles
TEST_CASES = [
    {
        "filename": "1_digital_arrest_scam.mp3",
        "voice": "hi-IN-MadhurNeural",
        "text": "Main Crime Branch Mumbai se Inspector Vijay Kumar bol raha hoon. Aapke Aadhaar card par ek parcel intercept hua hai jisme fake passports aur illegal drugs mile hain. You are under immediate digital arrest. Disconnect mat karna warna police aapke ghar pahunch jayegi."
    },
    {
        "filename": "2_stock_crypto_trap.mp3",
        "voice": "en-IN-PrabhatNeural",
        "text": "Congratulations sir! Guaranteed double returns in stock market trading. Invest money today for guaranteed profit. Immediate initial deposit fee of five thousand rupees required. Register now at free-crypto-airdrop.xyz before deadline."
    },
    {
        "filename": "3_ai_family_voice_clone.mp3",
        "voice": "en-US-AndrewNeural",
        "text": "Papa please help me! Mera accident ho gaya hai and police arrested me. Please send 50,000 rupees immediately to release me from the police station. Do not tell anyone else!"
    },
    {
        "filename": "4_remote_access_anydesk.mp3",
        "voice": "hi-IN-SwaraNeural",
        "text": "Dear consumer, your electricity power supply will be disconnected tonight at 9 PM due to pending KYC verification. Kindly install AnyDesk app on your mobile phone and share the 9 digit code immediately."
    },
    {
        "filename": "5_ceo_impersonation_wire.mp3",
        "voice": "en-US-BrianNeural",
        "text": "This is the Chief Executive Officer speaking. I am currently in a confidential acquisition meeting. I need you to execute an immediate offshore wire transfer of 250,000 dollars to our vendor account right now before close of business."
    }
]

async def compile_threat_voice_samples():
    output_dir = os.path.join(backend_dir, "test_audio_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    print("==========================================================")
    print("      LINGUASHIELD REAL-WORLD SCAM VOICE GENERATOR        ")
    print("==========================================================")
    
    for case in TEST_CASES:
        target_path = os.path.join(output_dir, case["filename"])
        print(f"[+] Compiling: {case['filename']}")
        print(f"    Voice Profile: {case['voice']}")
        
        try:
            communicator = edge_tts.Communicate(text=case["text"], voice=case["voice"])
            await communicator.save(target_path)
            print(f"[✓] File serialized: {target_path}")
        except Exception as e:
            print(f"[!] Compilation failed: {e}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(compile_threat_voice_samples())