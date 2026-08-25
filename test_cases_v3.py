import requests
import json
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tamil-llama"

# Recommended system prompt (from prompt engineering doc)
SYSTEM_PROMPT = """You are a Tamil grammar and writing assistant. Your task is to check Tamil input sentences for errors and correct them. You should identify and fix all grammatical, spelling, punctuation, and syntactic mistakes without changing the original meaning or tone. For example, correct verb tense, agreement, case-suffixes, gender forms, word order, and diacritics as needed. Always preserve the input's script and style. If the input is already grammatically correct, return it verbatim (unchanged). Do NOT hallucinate or add information not present in the input; focus only on grammar. Output only the corrected Tamil sentence (no quotes or extra explanation).

Tone: Polite, precise, and helpful.
Constraints: No translation (stay in Tamil), no creative rewriting. If the input contains harmful or disallowed content, refuse safely (e.g. say "மன்னிக்கவும், உதவி செய்ய முடியவில்லை.").

Example: User: "நான் நேற்று பள்ளிக்கு போகிறேன்." -> Assistant: "நான் நேற்று பள்ளிக்குப் போனேன்." """

# 40 Tamil grammar correction test cases across 10 categories
test_cases = [
    # --- TENSE (4 cases) ---
    ("Tense", "நான் நேற்று பள்ளிக்கு போகிறேன்.", "நான் நேற்று பள்ளிக்குப் போனேன்."),
    ("Tense", "அவள் நேற்று சந்தைக்கு செல்கிறாள்.", "அவள் நேற்று சந்தைக்குச் சென்றாள்."),
    ("Tense", "நான் கடந்த மாதம் திரைப்படம் பார்க்கிறேன்.", "நான் கடந்த மாதம் திரைப்படம் பார்த்தேன்."),
    ("Tense", "அவர்கள் இன்றைக்கு வேலைக்கு வருகிறார்கள்.", "அவர்கள் இன்றைக்கு வேலைக்கு வந்தனர்."),

    # --- AGREEMENT / S-V (4 cases) ---
    ("Agreement", "அவன் நல்ல பெண்.", "அவன் நல்ல பையன்."),
    ("Agreement", "அவள் நல்ல மாணவன்.", "அவள் நல்ல மாணவி."),
    ("Agreement", "நான் தமிழ் பேசுகிறது.", "நான் தமிழ் பேசுகிறேன்."),
    ("Agreement", "அவர்கள் நேற்று கிரிக்கெட் விளையாடுகிறான்.", "அவர்கள் நேற்று கிரிக்கெட் விளையாடினார்கள்."),

    # --- GENDER (4 cases) ---
    ("Gender", "இந்த பெண் ஒரு நல்ல மாணவன்.", "இந்த பெண் ஒரு நல்ல மாணவி."),
    ("Gender", "அவள் நல்ல தலைவன்.", "அவள் நல்ல தலைவர்."),
    ("Gender", "இவன் நல்ல அம்மா.", "அவள் நல்ல அம்மா."),
    ("Gender", "அவர் நல்ல மனிதர்.", "அவர் நல்ல பெண்."),

    # --- CASE MARKERS (4 cases) ---
    ("Case Markers", "அவன் வீட்டில் போனான்.", "அவன் வீட்டிற்கு சென்றான்."),
    ("Case Markers", "நான் நடக்கிறேன் நூலகம்.", "நான் நூலகத்திற்கு நடக்கிறேன்."),
    ("Case Markers", "அவள் தினம் அம்மாவை திட்டுகிறாள்.", "அவள் தினம் அம்மாவை விரும்புகிறாள்."),
    ("Case Markers", "அக்கா சாப்பிட்டுக் கொண்டாள் உறைகிறது.", "அக்கா சாப்பிட்டுக் கொண்டாள் உறங்குகிறாள்."),

    # --- WORD ORDER (4 cases) ---
    ("Word Order", "சமையல் அவன் செய்கிறான்.", "அவன் சமையல் செய்கிறான்."),
    ("Word Order", "ஆசிரியர் பாடம் மீண்டும் தொடங்கினார்.", "ஆசிரியர் மீண்டும் பாடத்தைத் தொடங்கினார்."),
    ("Word Order", "நாங்கள் நீர் குடிக்கிறார்.", "நாங்கள் நீர் குடித்தோம்."),
    ("Word Order", "அவள் பள்ளிக்கு சென்றாள் பிறகு வீட்டிற்கு வந்தாள்.", "அவள் பள்ளிக்கு சென்ற பிறகு வீட்டிற்கு வந்தாள்."),

    # --- MORPHOLOGY (4 cases) ---
    ("Morphology", "அவன் அழுகிறது.", "அவன் அழுகிறான்."),
    ("Morphology", "நான் சாப்பிட்டுக் கொண்டிருந்தான்.", "நான் சாப்பிட்டுக் கொண்டிருந்தேன்."),
    ("Morphology", "அந்தப் புத்தகம் ஆகிறதா?", "அந்தப் புத்தகம் ஆகிறதா?"),
    ("Morphology", "அவள் தமிழோடு பேசுகிறாள்.", "அவள் தமிழில் பேசுகிறாள்."),

    # --- REGISTER / COLLOQUIAL (4 cases) ---
    ("Register", "அவங்க எப்ப வர்றாங்க.", "அவர்கள் எப்போது வருவார்கள்."),
    ("Register", "நாங்க இல்ல போய் வந்தோம்.", "நாம் இல்லத்துக்கு போய் வந்தோம்."),
    ("Register", "அவள் சொன்னா நல்லு.", "அவள் சொன்னது நல்லது."),
    ("Register", "உங்களுக்கு என்ன ப்ராப்லம்?", "உங்களுக்கு என்ன பிரச்சனை?"),

    # --- CODE-MIXED / ENGLISH (4 cases) ---
    ("Code-mixed", "அவள் computer-ல் வேலை செய்கிறாள்.", "அவள் கணினியில் வேலை செய்கிறாள்."),
    ("Code-mixed", "நான் library-க்கு போனேன்.", "நான் நூலகத்திற்கு போனேன்."),
    ("Code-mixed", "அவர் meal சாப்பிட்டார்.", "அவர் உணவு சாப்பிட்டார்."),
    ("Code-mixed", "அவன் TV பார்த்து தூங்கினான்.", "அவன் தொலைக்காட்சியில் பார்த்து தூங்கினான்."),

    # --- PUNCTUATION (4 cases) ---
    ("Punctuation", "அவள் பாடம் முடிந்ததும், அவர் வெளியே சென்றார்.", "அவள் பாடம் முடிந்ததும் அவர் வெளியே சென்றார்."),
    ("Punctuation", "நான் வருகிறேன் நான்.", "நான் வருகிறேன்."),
    ("Punctuation", "அவள் எங்கே போகிறாள்?", "அவள் எங்கே போகிறாள்?"),
    ("Punctuation", "அக்கா அழகு இருக்கும் 🙂", "அக்கா அழகு இருக்கிறாள்."),

    # --- AMBIGUOUS / OTHER (4 cases) ---
    ("Ambiguous", "வானம் உயரம்.", "வானம் உயரமானது."),
    ("Ambiguous", "அவர் நிகழ்ச்சி.", "அவர் நிகழ்ச்சியில் கலந்து கொண்டார்."),
    ("Ambiguous", "நண்பன் அவன்.", "அவன் நண்பன்."),
    ("Ambiguous", "நான் உண்ணினேன் நீ?", "நான் உண்ணினேன். நீ?"),
]

# Counters
results = {cat: {"pass": 0, "fail": 0} for cat in set(tc[0] for tc in test_cases)}
total_pass = 0
total_fail = 0

print("=" * 90)
print("TAMIL GRAMMAR CORRECTION TEST SUITE — 40 Cases (10 Categories)")
print("=" * 90)

for i, (category, wrong, expected) in enumerate(test_cases, 1):
    prompt = f"திருத்தப்பட்ட உரை:\n{wrong}"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1},
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json().get("response", "NO RESPONSE").strip()
    except Exception as e:
        result = f"ERROR: {e}"

    # Check match
    match = expected.strip() in result
    status = "PASS" if match else "FAIL"

    if match:
        results[category]["pass"] += 1
        total_pass += 1
    else:
        results[category]["fail"] += 1
        total_fail += 1

    print(f"\nTest #{i:02d} [{category}] — {status}")
    print(f"  Input:    {wrong}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")

# Summary by category
print("\n" + "=" * 90)
print("RESULTS BY CATEGORY")
print("=" * 90)
for cat in sorted(results.keys()):
    p = results[cat]["pass"]
    f = results[cat]["fail"]
    total = p + f
    print(f"  {cat:<20s}  {p}/{total} passed  ({'100%' if f == 0 else f'{p/total*100:.0f}%'})")

print("\n" + "=" * 90)
print(f"OVERALL: {total_pass} passed, {total_fail} failed out of {len(test_cases)}")
print(f"ACCURACY: {total_pass/len(test_cases)*100:.1f}%")
print("=" * 90)
