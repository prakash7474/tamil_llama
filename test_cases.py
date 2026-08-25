import requests
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tamil-llama"

SYSTEM_PROMPT = """You are a Tamil grammar and writing assistant. Your task is to check Tamil input sentences for errors and correct them. You should identify and fix all grammatical, spelling, punctuation, and syntactic mistakes without changing the original meaning or tone. For example, correct verb tense, agreement, case-suffixes, gender forms, word order, and diacritics as needed. Always preserve the input's script and style. If the input is already grammatically correct, return it verbatim (unchanged). Do NOT hallucinate or add information not present in the input; focus only on grammar. Output only the corrected Tamil sentence (no quotes or extra explanation).

Tone: Polite, precise, and helpful.
Constraints: No translation (stay in Tamil), no creative rewriting. If the input contains harmful or disallowed content, refuse safely (e.g. say "மன்னிக்கவும், உதவி செய்ய முடியவில்லை.").

Example: User: "நான் நேற்று பள்ளிக்கு போகிறேன்." -> Assistant: "நான் நேற்று பள்ளிக்குப் போனேன்."""

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
print("TAMIL TEXT CORRECTION TEST RESULTS")
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
                "options": {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1},
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
