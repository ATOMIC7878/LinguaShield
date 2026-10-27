from backend.localization.translation_ops import translate_content_to_english

tests = {
    "HI-1": "आपका बैंक खाता बंद होने वाला है। कृपया तुरंत अपना ओटीपी साझा करें।",
    "HI-2": "आपको डिजिटल अरेस्ट के तहत रखा गया है और निर्दोष साबित करने के लिए पैसे ट्रांसफर करने होंगे।",
    "HI-3": "यह एक सुरक्षा जागरूकता संदेश है। अपना ओटीपी किसी के साथ साझा न करें।",

    "MR-1": "तुमचे बँक खाते बंद होणार आहे. कृपया तुमचा ओटीपी ताबडतोब शेअर करा.",
    "MR-2": "तुम्हाला डिजिटल अरेस्टखाली ठेवण्यात आले आहे आणि निर्दोष सिद्ध करण्यासाठी पैसे ट्रान्सफर करावे लागतील.",
    "MR-3": "हा एक सुरक्षा जागरूकता संदेश आहे. तुमचा ओटीपी कोणाशीही शेअर करू नका."
}

print("=" * 90)
print("LINGUASHIELD - NLLB TRANSLATION CROSS-CHECK")
print("=" * 90)

for name, text in tests.items():
    print("-" * 90)
    print(name)
    print("INPUT :", text)

    try:
        result = translate_content_to_english(text)
        print("OUTPUT:", result)
    except Exception as e:
        print("ERROR :", e)

print("=" * 90)
print("TRANSLATION CHECK COMPLETE")
print("=" * 90)
