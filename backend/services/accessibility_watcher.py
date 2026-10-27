import logging
from typing import Dict, List, Any

logger = logging.getLogger("LinguaShield.AccessibilityWatcher")

# Recognized High-Risk Remote Administration & Screen-Sharing App Packages
KNOWN_RAT_PACKAGES = {
    "com.anydesk.anydeskandroid": "AnyDesk Remote Control",
    "com.teamviewer.quicksupport.market": "TeamViewer QuickSupport",
    "com.teamviewer.host.market": "TeamViewer Host",
    "com.rustdesk.rustdesk": "RustDesk Remote",
    "com.splashtop.remote": "Splashtop Remote",
    "com.zoho.assist": "Zoho Assist",
    "com.airdroid": "AirDroid Remote Access",
    "com.vysor.desktop": "Vysor Screen Control"
}

class AccessibilityAppWatcher:
    """
    On-Device Accessibility App & Remote Control Watcher for LinguaShield.
    Detects active AnyDesk/TeamViewer sessions, accessibility service abuse,
    and concurrent screen projection during telephony calls.
    """
    def __init__(self):
        self.rat_signatures = KNOWN_RAT_PACKAGES

    def evaluate_device_telemetry(self, telemetry_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses device telemetry provided by client agents (Android/iOS endpoints).
        Non-destructive, returning a clean threat score vector.
        """
        if not telemetry_payload:
            return {
                "rat_detected": False,
                "threat_score": 0.0,
                "detected_packages": [],
                "active_accessibility_services": [],
                "screen_sharing_active": False
            }

        running_packages = [pkg.lower() for pkg in telemetry_payload.get("running_packages", [])]
        enabled_accessibility = [acc.lower() for acc in telemetry_payload.get("enabled_accessibility_services", [])]
        is_screen_sharing = bool(telemetry_payload.get("is_screen_sharing", False))
        overlay_active = bool(telemetry_payload.get("accessibility_overlay_detected", False))

        detected_rat_apps = []
        
        # Check active packages against RAT registry
        for pkg, label in self.rat_signatures.items():
            if any(pkg in r_pkg for r_pkg in running_packages) or any(pkg in a_svc for a_svc in enabled_accessibility):
                detected_rat_apps.append({"package": pkg, "name": label})

        threat_score = 0.0
        
        # Scoring Logic:
        # 1. Active RAT app running or enabled in accessibility (+0.60)
        if detected_rat_apps:
            threat_score += 0.60
            
        # 2. Active Screen Projection / Mirroring (+0.25)
        if is_screen_sharing:
            threat_score += 0.25
            
        # 3. Unsanctioned Accessibility Overlay Active (+0.25)
        if overlay_active:
            threat_score += 0.25

        threat_score = min(1.0, round(threat_score, 4))
        is_rat_active = threat_score >= 0.70 or (len(detected_rat_apps) > 0 and is_screen_sharing)

        logger.info(f"[Accessibility Watcher] RAT Risk Score: {threat_score * 100:.1f}% | Active RAT Apps: {len(detected_rat_apps)}")

        return {
            "rat_detected": is_rat_active,
            "threat_score": round(threat_score * 100, 2),
            "detected_packages": detected_rat_apps,
            "screen_sharing_active": is_screen_sharing,
            "accessibility_overlay_active": overlay_active,
            "classification_label": "REMOTE_ACCESS_TAKEOVER_DETECTED" if is_rat_active else "NO_RAT_ACTIVE"
        }