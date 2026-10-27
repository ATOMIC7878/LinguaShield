import io
import json
import requests
from PIL import Image, ImageDraw, ImageFont

def run_live_pipeline_test():
    """Generates a high-fidelity mock screenshot canvas using native system fonts 
    and standard spacing to verify end-to-end multi-stage URL recovery.
    """
    print("[1] Generating high-resolution mock screenshot canvas in memory...")
    
    # Clean, expansive canvas boundary
    image_width = 1200
    image_height = 450
    img = Image.new('RGB', (image_width, image_height), color=(255, 255, 255))
    canvas = ImageDraw.Draw(img)
    
    # Standard single-spacing layout. Crisp Arial 28pt prevents character bleeding 
    # naturally while keeping bounding boxes tight enough for spatial welding.
    scam_text = (
        "SYSTEM ALERT: Unusual login activity detected on your account.\n"
        "Verify your profile credentials immediately to stop suspension:\n\n"
        "http www refund tax gov xyz login"
    )
    
    # Load native Windows Arial font
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        print("[✓] Native Windows Arial font loaded successfully.")
    except Exception as font_load_error:
        print(f"[!] System font lookup variance: {font_load_error}. Using fallback initialization.")
        font = None
    
    # Draw the text payload cleanly
    canvas.text((60, 60), scam_text, fill=(0, 0, 0), font=font)
    
    # Save the generated image directly to a byte buffer
    byte_stream = io.BytesIO()
    img.save(byte_stream, format='PNG')
    image_bytes = byte_stream.getvalue()
    
    print("[2] Shipping binary image stream to FastAPI endpoint...")
    target_api_url = "http://127.0.0.1:8000/api/analyze/file"
    multipart_form_data = {
        'file': ('demo_phishing_screenshot.png', image_bytes, 'image/png')
    }
    
    try:
        response = requests.post(target_api_url, files=multipart_form_data)
        
        print("\n" + "="*50)
        print("🔍 LINGUASHIELD LIVE THREAT ASSESSMENT REPORT")
        print("="*50)
        print(json.dumps(response.json(), indent=4))
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("\n[!] Connection Error: FastAPI server could not be reached.")
        print("Please verify that 'python backend/main.py' is currently running on port 8000.")
    except Exception as error:
        print(f"\n[!] Pipeline Execution Interruption: {error}")

if __name__ == "__main__":
    run_live_pipeline_test()