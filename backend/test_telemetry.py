import os
import sys

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.accessibility_watcher import AccessibilityAppWatcher

def run_telemetry_tests():
    watcher = AccessibilityAppWatcher()
    
    print("==========================================================")
    print("     LINGUASHIELD ACCESSIBILITY / RAT WATCHER TEST        ")
    print("==========================================================")
    
    # Scenario A: AnyDesk Active + Screen Sharing (High Risk Takeover)
    scam_payload = {
        "running_packages": ["com.anydesk.anydeskandroid", "com.whatsapp"],
        "enabled_accessibility_services": ["com.anydesk.anydeskandroid/AccessibilityService"],
        "is_screen_sharing": True,
        "accessibility_overlay_detected": True
    }
    
    # Scenario B: Normal Clean Device State
    clean_payload = {
        "running_packages": ["com.google.android.youtube", "com.whatsapp"],
        "enabled_accessibility_services": [],
        "is_screen_sharing": False,
        "accessibility_overlay_detected": False
    }
    
    print("\n[+] TEST CASE 1: AnyDesk Remote Desktop Takeover Attempt")
    report_a = watcher.evaluate_device_telemetry(scam_payload)
    print(f" • RAT Threat Detected : {report_a['rat_detected']}")
    print(f" • Threat Confidence   : {report_a['threat_score']}%")
    print(f" • Classification Tag  : {report_a['classification_label']}")
    print(f" • Detected RAT Apps   : {report_a['detected_packages']}")
    
    print("\n[+] TEST CASE 2: Clean Device Activity")
    report_b = watcher.evaluate_device_telemetry(clean_payload)
    print(f" • RAT Threat Detected : {report_b['rat_detected']}")
    print(f" • Threat Confidence   : {report_b['threat_score']}%")
    print(f" • Classification Tag  : {report_b['classification_label']}")
    print("==========================================================")

if __name__ == "__main__":
    run_telemetry_tests()