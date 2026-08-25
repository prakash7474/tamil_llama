import requests
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tamil-llama"

# Much more specific prompt with grammar rules
SYSTEM_PROMPT = """நீங்கள் ஒரு தமிழ் இலக்கண நிபுணர். பயனர் அனுப்பும் தமிழ் உரையில் உள்ள இலக்கணப் பிழைகளை மட்டும் சரிசெய்து, சரியான உரையை மட்டும் திருப்பி அளிக்கவும்.

முக்கிய தமிழ் இலக்கண விதிகள்:
1. கருத்துரு முரண்: "நான் ... கிறேன்" (முதல் நபர்), "அவன் ... கிறான்" (ஆண்), "அவள் ... கிறாள்" (பெண்), "அவர்கள் ... கிறார்கள்" (பன்மை)
2. கால முரண்: "நேற்று" என்றால் கடந்த காலம் பயன்படுத்த வேண்டும் (போனேன், சென்றான், வந்தாள்)
3. பால் முரண்: ஆண் நபருக்கு "அவன்/பையன்/மாணவன்", பெண் நபருக்கு "அவள்/பெண்/மாணவி"
4. உரிமை முரண்: "எனக்கு" பதில் "என்னிடம்" பயன்படுத்த வேண்டும் (பொருள் குறிப்பிடும் போது)
5. இணைப்பு முரண்: வன்மை முடிவு: "பள்ளிக்குப் போனேன்" (இணைப்பு "ப்" பயன்படுத்த வேண்டும்)

விதிகள்:
- சரியான உரையை மட்டும் திருப்பி அளிக்கவும்
- விளக்கம் வேண்டாம், திருத்தப்பட்ட வாக்கியம் மட்டும் போதும்
- உரையின் அர்த்தத்தை மாற்றாதீர்கள்"""

test_cases = [
    ("நான் நேற்று பள்ளிக்கு போகிறேன்.", "நான் நேற்று பள்ளிக்குப் போனேன்."),
    ("அவன் தினமும் பள்ளிக்கு சென்றான்.", "அவன் தினமும் பள்ளிக்குச் செல்கிறான்."),
    ("நான் தமிழ் பேசுகிறது.", "நான் தமிழ் பேசுகிறேன்."),
    ("அவள் நேற்று சந்தைக்கு செல்கிறாள்.", "அவள் நேற்று சந்தைக்குச் சென்றாள்."),
    ("அவர்கள் நேற்று கிரிக்கெட் விளையாடுகிறான்.", "அவர்கள் நேற்று கிரிக்கெட் விளையாடினார்கள்."),
    ("எனக்கு ஒரு புத்தகம் இருக்கிறான்.", "என்னிடம் ஒரு புத்தகம் இருக்கிறது."),
    ("அவன் நல்ல பெண்.", "அவன் நல்ல பையன்."),
    ("அவள் நல்ல மாணவன்.", "அவள் நல்ல மாணவி."),
    ("நான் உணவு சாப்பிட்டுக் கொண்டிருந்தான்.", "நான் உணவு சாப்பிட்டுக் கொண்டிருந்தேன்."),
    ("ரவி மற்றும் குமார் பள்ளிக்கு சென்றான்.", "ரவி மற்றும் குமார் பள்ளிக்குச் சென்றார்கள்."),
]

print("=" * 80)
print("TAMIL TEXT CORRECTION TEST RESULTS (Improved Prompt)")
print("=" * 80)

pass_count = 0
fail_count = 0

for i, (wrong, expected) in enumerate(test_cases, 1):
    prompt = f"திருத்தப்பட்ட உரை:\n{wrong}"
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": 0.1, "repeat_penalty": 1.1},
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json().get("response", "NO RESPONSE").strip()
    except Exception as e:
        result = f"ERROR: {e}"
    
    # Check if expected text appears in the response
    match = expected.strip() in result
    status = "PASS" if match else "FAIL"
    
    if match:
        pass_count += 1
    else:
        fail_count += 1
    
    print(f"\nTest #{i}: {status}")
    print(f"  Input:    {wrong}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")

print("\n" + "=" * 80)
print(f"RESULTS: {pass_count} passed, {fail_count} failed out of {len(test_cases)}")
print("=" * 80)
