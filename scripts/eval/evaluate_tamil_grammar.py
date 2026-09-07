#!/usr/bin/env python3
"""
Tamil Grammar Correction Evaluation Script
============================================

Evaluates a model's Tamil grammar correction ability using:
  1. Curated test cases (40 cases, 10 categories) with exact matching
  2. eval.jsonl dataset with fuzzy/semantic similarity metrics
  3. Optional GPT-4 scoring for quality assessment

Supports HuggingFace models (QLoRA/fine-tuned) and Ollama.

Usage:
    # Evaluate a HuggingFace model on test cases
    python evaluate_tamil_grammar.py --model Qwen/Qwen3-8B --mode test-cases

    # Evaluate on eval.jsonl with metrics
    python evaluate_tamil_grammar.py --model Qwen/Qwen3-8B --mode eval-set --eval-file eval.jsonl

    # Evaluate an Ollama model
    python evaluate_tamil_grammar.py --ollama tamil-llama --mode test-cases

    # Full evaluation (both test cases + eval set)
    python evaluate_tamil_grammar.py --model ./merged_model --mode all --eval-file eval.jsonl

    # Compare two models
    python evaluate_tamil_grammar.py --model model_a --mode test-cases --compare-with model_b
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Tamil test cases (from test_cases_v3.py, expanded)
# ============================================================

TEST_CASES = [
    # --- TENSE ---
    ("Tense", "நான் நேற்று பள்ளிக்கு போகிறேன்.", "நான் நேற்று பள்ளிக்குப் போனேன்."),
    ("Tense", "அவள் நேற்று சந்தைக்கு செல்கிறாள்.", "அவள் நேற்று சந்தைக்குச் சென்றாள்."),
    ("Tense", "நான் கடந்த மாதம் திரைப்படம் பார்க்கிறேன்.", "நான் கடந்த மாதம் திரைப்படம் பார்த்தேன்."),
    ("Tense", "அவர்கள் இன்றைக்கு வேலைக்கு வருகிறார்கள்.", "அவர்கள் இன்றைக்கு வேலைக்கு வந்தனர்."),

    # --- AGREEMENT ---
    ("Agreement", "அவன் நல்ல பெண்.", "அவன் நல்ல பையன்."),
    ("Agreement", "அவள் நல்ல மாணவன்.", "அவள் நல்ல மாணவி."),
    ("Agreement", "நான் தமிழ் பேசுகிறது.", "நான் தமிழ் பேசுகிறேன்."),
    ("Agreement", "அவர்கள் நேற்று கிரிக்கெட் விளையாடுகிறான்.", "அவர்கள் நேற்று கிரிக்கெட் விளையாடினார்கள்."),

    # --- GENDER ---
    ("Gender", "இந்த பெண் ஒரு நல்ல மாணவன்.", "இந்த பெண் ஒரு நல்ல மாணவி."),
    ("Gender", "அவள் நல்ல தலைவன்.", "அவள் நல்ல தலைவர்."),
    ("Gender", "இவன் நல்ல அம்மா.", "அவள் நல்ல அம்மா."),
    ("Gender", "அவர் நல்ல மனிதர்.", "அவர் நல்ல பெண்."),

    # --- CASE MARKERS ---
    ("Case Markers", "அவன் வீட்டில் போனான்.", "அவன் வீட்டிற்கு சென்றான்."),
    ("Case Markers", "நான் நடக்கிறேன் நூலகம்.", "நான் நூலகத்திற்கு நடக்கிறேன்."),
    ("Case Markers", "அவள் தினம் அம்மாவை திட்டுகிறாள்.", "அவள் தினம் அம்மாவை விரும்புகிறாள்."),
    ("Case Markers", "அக்கா சாப்பிட்டுக் கொண்டாள் உறைகிறது.", "அக்கா சாப்பிட்டுக் கொண்டாள் உறங்குகிறாள்."),

    # --- WORD ORDER ---
    ("Word Order", "சமையல் அவன் செய்கிறான்.", "அவன் சமையல் செய்கிறான்."),
    ("Word Order", "ஆசிரியர் பாடம் மீண்டும் தொடங்கினார்.", "ஆசிரியர் மீண்டும் பாடத்தைத் தொடங்கினார்."),
    ("Word Order", "நாங்கள் நீர் குடிக்கிறார்.", "நாங்கள் நீர் குடித்தோம்."),
    ("Word Order", "அவள் பள்ளிக்கு சென்றாள் பிறகு வீட்டிற்கு வந்தாள்.", "அவள் பள்ளிக்கு சென்ற பிறகு வீட்டிற்கு வந்தாள்."),

    # --- MORPHOLOGY ---
    ("Morphology", "அவன் அழுகிறது.", "அவன் அழுகிறான்."),
    ("Morphology", "நான் சாப்பிட்டுக் கொண்டிருந்தான்.", "நான் சாப்பிட்டுக் கொண்டிருந்தேன்."),
    ("Morphology", "அந்தப் புத்தகம் ஆகிறதா?", "அந்தப் புத்தகம் ஆகிறதா?"),
    ("Morphology", "அவள் தமிழோடு பேசுகிறாள்.", "அவள் தமிழில் பேசுகிறாள்."),

    # --- REGISTER ---
    ("Register", "அவங்க எப்ப வர்றாங்க.", "அவர்கள் எப்போது வருவார்கள்."),
    ("Register", "நாங்க இல்ல போய் வந்தோம்.", "நாம் இல்லத்துக்கு போய் வந்தோம்."),
    ("Register", "அவள் சொன்னா நல்லு.", "அவள் சொன்னது நல்லது."),
    ("Register", "உங்களுக்கு என்ன ப்ராப்லம்?", "உங்களுக்கு என்ன பிரச்சனை?"),

    # --- CODE-MIXED ---
    ("Code-mixed", "அவள் computer-ல் வேலை செய்கிறாள்.", "அவள் கணினியில் வேலை செய்கிறாள்."),
    ("Code-mixed", "நான் library-க்கு போனேன்.", "நான் நூலகத்திற்கு போனேன்."),
    ("Code-mixed", "அவர் meal சாப்பிட்டார்.", "அவர் உணவு சாப்பிட்டார்."),
    ("Code-mixed", "அவன் TV பார்த்து தூங்கினான்.", "அவன் தொலைக்காட்சியில் பார்த்து தூங்கினான்."),

    # --- PUNCTUATION ---
    ("Punctuation", "அவள் பாடம் முடிந்ததும், அவர் வெளியே சென்றார்.", "அவள் பாடம் முடிந்ததும் அவர் வெளியே சென்றார்."),
    ("Punctuation", "நான் வருகிறேன் நான்.", "நான் வருகிறேன்."),
    ("Punctuation", "அவள் எங்கே போகிறாள்?", "அவள் எங்கே போகிறாள்?"),
    ("Punctuation", "அக்கா அழகு இருக்கும் 🙂", "அக்கா அழகு இருக்கிறாள்."),

    # --- AMBIGUOUS ---
    ("Ambiguous", "வானம் உயரம்.", "வானம் உயரமானது."),
    ("Ambiguous", "அவர் நிகழ்ச்சி.", "அவர் நிகழ்ச்சியில் கலந்து கொண்டார்."),
    ("Ambiguous", "நண்பன் அவன்.", "அவன் நண்பன்."),
    ("Ambiguous", "நான் உண்ணினேன் நீ?", "நான் உண்ணினேன். நீ?"),
]

TAMIL_RANGE = (0x0B80, 0x0BFF)


# ============================================================
# Metrics
# ============================================================

def is_tamil_char(ch):
    cp = ord(ch)
    return TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1]


def normalize_tamil(text):
    """Normalize Tamil text for comparison."""
    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove punctuation for comparison
    text = re.sub(r"[.,!?;:\u0BCD\u0BD7\u0BF4]", "", text)
    # Normalize common Tamil character variations
    text = text.replace("\u0BCD", "")  # Remove virama for comparison
    return text.lower().strip()


def exact_match(predicted, expected):
    """Check exact match after normalization."""
    return normalize_tamil(predicted) == normalize_tamil(expected)


def fuzzy_match(predicted, expected, threshold=0.8):
    """Check fuzzy match using SequenceMatcher."""
    norm_pred = normalize_tamil(predicted)
    norm_exp = normalize_tamil(expected)
    ratio = SequenceMatcher(None, norm_pred, norm_exp).ratio()
    return ratio >= threshold, ratio


def token_overlap(predicted, expected):
    """Compute token-level overlap (Jaccard similarity)."""
    pred_tokens = set(normalize_tamil(predicted).split())
    exp_tokens = set(normalize_tamil(expected).split())
    if not exp_tokens:
        return 0.0
    intersection = pred_tokens & exp_tokens
    union = pred_tokens | exp_tokens
    return len(intersection) / len(union) if union else 0.0


def character_overlap(predicted, expected):
    """Compute character-level overlap."""
    pred_chars = set(normalize_tamil(predicted))
    exp_chars = set(normalize_tamil(expected))
    if not exp_chars:
        return 0.0
    intersection = pred_chars & exp_chars
    union = pred_chars | exp_chars
    return len(intersection) / len(union) if union else 0.0


def word_error_rate(predicted, expected):
    """Compute word error rate (lower is better)."""
    pred_words = normalize_tamil(predicted).split()
    exp_words = normalize_tamil(expected).split()
    if not exp_words:
        return 1.0 if pred_words else 0.0
    # Simple WER using edit distance
    d = [[0] * (len(exp_words) + 1) for _ in range(len(pred_words) + 1)]
    for i in range(len(pred_words) + 1):
        d[i][0] = i
    for j in range(len(exp_words) + 1):
        d[0][j] = j
    for i in range(1, len(pred_words) + 1):
        for j in range(1, len(exp_words) + 1):
            if pred_words[i - 1] == exp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[len(pred_words)][len(exp_words)] / len(exp_words)


def detect_repetition(text):
    """Detect if the model output is repetitive."""
    words = text.split()
    if len(words) < 5:
        return False, 0.0
    # Check for repeated n-grams
    for n in [3, 4]:
        if len(words) < n * 2:
            continue
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)
        for ngram, count in ngram_counts.items():
            if count >= 3:
                return True, count / len(ngrams)
    return False, 0.0


def is_tamil_dominant(text, threshold=0.3):
    """Check if text is predominantly Tamil."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    tamil_count = sum(1 for c in chars if is_tamil_char(c))
    return (tamil_count / len(chars)) >= threshold


# ============================================================
# Model inference
# ============================================================

def load_hf_model(model_name, device="auto"):
    """Load a HuggingFace model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if model needs quantization
    if Path(model_name).exists() or "/" not in model_name:
        # Local model - load directly
        print(f"Loading model from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        # Remote model - try 4-bit first
        print(f"Loading model {model_name} in 4-bit...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except Exception:
            print("4-bit failed, loading in float16...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )

    model.eval()
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


def generate_hf(model, tokenizer, prompt, system_prompt=None, max_new_tokens=200):
    """Generate text using a HuggingFace model."""
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: just use the prompt directly
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def generate_ollama(model_name, prompt, system_prompt=None, url="http://localhost:11434/api/generate"):
    """Generate text using Ollama API."""
    import requests

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.1},
    }
    if system_prompt:
        payload["system"] = system_prompt

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


# ============================================================
# Evaluation
# ============================================================

SYSTEM_PROMPT = (
    "You are a Tamil grammar and writing assistant. Your task is to check Tamil "
    "input sentences for errors and correct them. Identify and fix all grammatical, "
    "spelling, and syntactic mistakes without changing the original meaning. "
    "Output only the corrected Tamil sentence."
)


def evaluate_test_cases(generator_fn, cases=TEST_CASES):
    """Evaluate on curated test cases."""
    results = []
    for i, (category, wrong, expected) in enumerate(cases):
        prompt = f"திருத்தப்பட்ட உரை:\n{wrong}"
        try:
            predicted = generator_fn(prompt)
        except Exception as e:
            predicted = f"ERROR: {e}"

        em = exact_match(predicted, expected)
        fm_score, fm_ratio = fuzzy_match(predicted, expected)
        to_score = token_overlap(predicted, expected)
        co_score = character_overlap(predicted, expected)
        is_rep, rep_ratio = detect_repetition(predicted)
        is_tamil = is_tamil_dominant(predicted)

        results.append({
            "category": category,
            "input": wrong,
            "expected": expected,
            "predicted": predicted,
            "exact_match": em,
            "fuzzy_match": fm_score,
            "fuzzy_ratio": fm_ratio,
            "token_overlap": to_score,
            "char_overlap": co_score,
            "is_repetitive": is_rep,
            "is_tamil": is_tamil,
        })

    return results


def evaluate_eval_set(generator_fn, eval_file, max_samples=None):
    """Evaluate on eval.jsonl dataset."""
    items = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if max_samples:
        items = items[:max_samples]

    results = []
    for i, item in enumerate(items):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        expected_output = item.get("output", "")

        # Extract the corrected sentence from expected output
        expected_correction = expected_output
        for marker in ["சரியான வாக்கியம்:", "சரியான வாக்கியம் :"]:
            if marker in expected_correction:
                expected_correction = expected_correction.split(marker)[1].split("\n")[0].strip()
                break

        prompt = f"{instruction}\n\n{input_text}" if input_text else instruction

        try:
            predicted = generator_fn(prompt)
        except Exception as e:
            predicted = f"ERROR: {e}"

        # Extract correction from predicted
        pred_correction = predicted
        for marker in ["சரியான வாக்கியம்:", "சரியான வாக்கியம் :"]:
            if marker in pred_correction:
                pred_correction = pred_correction.split(marker)[1].split("\n")[0].strip()
                break

        em = exact_match(pred_correction, expected_correction)
        fm_score, fm_ratio = fuzzy_match(pred_correction, expected_correction)
        to_score = token_overlap(pred_correction, expected_correction)
        co_score = character_overlap(pred_correction, expected_correction)
        is_rep, rep_ratio = detect_repetition(predicted)
        is_tamil = is_tamil_dominant(predicted)

        results.append({
            "index": i,
            "input": input_text,
            "expected": expected_correction,
            "predicted": predicted,
            "pred_correction": pred_correction,
            "exact_match": em,
            "fuzzy_match": fm_score,
            "fuzzy_ratio": fm_ratio,
            "token_overlap": to_score,
            "char_overlap": co_score,
            "is_repetitive": is_rep,
            "is_tamil": is_tamil,
        })

    return results


def compute_summary(results):
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    exact_matches = sum(1 for r in results if r["exact_match"])
    fuzzy_matches = sum(1 for r in results if r["fuzzy_match"])
    avg_fuzzy = sum(r["fuzzy_ratio"] for r in results) / n
    avg_token = sum(r["token_overlap"] for r in results) / n
    avg_char = sum(r["char_overlap"] for r in results) / n
    repetitions = sum(1 for r in results if r["is_repetitive"])
    tamil_dominant = sum(1 for r in results if r["is_tamil"])

    return {
        "total": n,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / n,
        "fuzzy_matches": fuzzy_matches,
        "fuzzy_match_rate": fuzzy_matches / n,
        "avg_fuzzy_ratio": avg_fuzzy,
        "avg_token_overlap": avg_token,
        "avg_char_overlap": avg_char,
        "repetitions": repetitions,
        "repetition_rate": repetitions / n,
        "tamil_dominant": tamil_dominant,
        "tamil_rate": tamil_dominant / n,
    }


def compute_category_summary(results):
    """Compute per-category summaries for test cases."""
    categories = defaultdict(list)
    for r in results:
        categories[r["category"]].append(r)

    summaries = {}
    for cat, cat_results in categories.items():
        n = len(cat_results)
        em = sum(1 for r in cat_results if r["exact_match"])
        fm = sum(1 for r in cat_results if r["fuzzy_match"])
        avg_f = sum(r["fuzzy_ratio"] for r in cat_results) / n
        summaries[cat] = {
            "total": n,
            "exact_matches": em,
            "exact_match_rate": em / n,
            "fuzzy_matches": fm,
            "fuzzy_match_rate": fm / n,
            "avg_fuzzy_ratio": avg_f,
        }
    return summaries


# ============================================================
# Report generation
# ============================================================

def generate_report(
    test_results,
    eval_results,
    test_summary,
    eval_summary,
    cat_summaries,
    output_path=None,
):
    """Generate a detailed evaluation report."""
    lines = []
    w = lines.append

    w("=" * 70)
    w("TAMIL GRAMMAR CORRECTION EVALUATION REPORT")
    w("=" * 70)
    w("")

    # Test cases results
    if test_results:
        w("--- TEST CASES RESULTS ---")
        w(f"  Total test cases:        {test_summary['total']}")
        w(f"  Exact matches:           {test_summary['exact_matches']}/{test_summary['total']} "
          f"({test_summary['exact_match_rate']:.1%})")
        w(f"  Fuzzy matches (>0.8):    {test_summary['fuzzy_matches']}/{test_summary['total']} "
          f"({test_summary['fuzzy_match_rate']:.1%})")
        w(f"  Avg fuzzy ratio:         {test_summary['avg_fuzzy_ratio']:.3f}")
        w(f"  Avg token overlap:       {test_summary['avg_token_overlap']:.3f}")
        w(f"  Avg char overlap:        {test_summary['avg_char_overlap']:.3f}")
        w(f"  Repetitive outputs:      {test_summary['repetitions']}/{test_summary['total']} "
          f"({test_summary['repetition_rate']:.1%})")
        w(f"  Tamil-dominant outputs:   {test_summary['tamil_dominant']}/{test_summary['total']} "
          f"({test_summary['tamil_rate']:.1%})")
        w("")

        # Per-category breakdown
        w("  Per-category breakdown:")
        w(f"  {'Category':<20s} {'EM':>5s} {'FM':>5s} {'Avg F':>8s}")
        w("  " + "-" * 42)
        for cat, cs in sorted(cat_summaries.items()):
            w(f"  {cat:<20s} {cs['exact_matches']}/{cs['total']:>2d}  "
              f"{cs['fuzzy_matches']}/{cs['total']:>2d}  "
              f"{cs['avg_fuzzy_ratio']:.3f}")
        w("")

        # Failed test cases
        failed = [r for r in test_results if not r["exact_match"]]
        if failed:
            w(f"  Failed test cases ({len(failed)}):")
            for r in failed[:10]:
                w(f"    [{r['category']}] Input: {r['input'][:50]}")
                w(f"      Expected: {r['expected'][:60]}")
                w(f"      Got:      {r['predicted'][:60]}")
                w(f"      Fuzzy:    {r['fuzzy_ratio']:.3f}")
            if len(failed) > 10:
                w(f"    ... and {len(failed) - 10} more")
        w("")

    # Eval set results
    if eval_results:
        w("--- EVAL SET RESULTS ---")
        w(f"  Total examples:          {eval_summary['total']}")
        w(f"  Exact matches:           {eval_summary['exact_matches']}/{eval_summary['total']} "
          f"({eval_summary['exact_match_rate']:.1%})")
        w(f"  Fuzzy matches (>0.8):    {eval_summary['fuzzy_matches']}/{eval_summary['total']} "
          f"({eval_summary['fuzzy_match_rate']:.1%})")
        w(f"  Avg fuzzy ratio:         {eval_summary['avg_fuzzy_ratio']:.3f}")
        w(f"  Avg token overlap:       {eval_summary['avg_token_overlap']:.3f}")
        w(f"  Avg char overlap:        {eval_summary['avg_char_overlap']:.3f}")
        w(f"  Repetitive outputs:      {eval_summary['repetitions']}/{eval_summary['total']} "
          f"({eval_summary['repetition_rate']:.1%})")
        w(f"  Tamil-dominant outputs:   {eval_summary['tamil_dominant']}/{eval_summary['total']} "
          f"({eval_summary['tamil_rate']:.1%})")
        w("")

        # Fuzzy ratio distribution
        ratios = [r["fuzzy_ratio"] for r in eval_results]
        bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        w("  Fuzzy ratio distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for r in ratios if bins[i] <= r < bins[i + 1])
            bar = "#" * min(count, 50)
            w(f"    [{bins[i]:.2f}-{bins[i+1]:.2f}): {count:>4d} {bar}")
        w("")

    # Overall assessment
    w("=" * 70)
    w("OVERALL ASSESSMENT")
    w("=" * 70)

    # Determine pass/fail based on project success criteria
    checks = []
    if test_results:
        em_rate = test_summary["exact_match_rate"]
        fm_rate = test_summary["fuzzy_match_rate"]
        tamil_rate = test_summary["tamil_rate"]
        rep_rate = test_summary["repetition_rate"]

        checks.append(("Exact match rate >= 50%", em_rate >= 0.50, f"{em_rate:.1%}"))
        checks.append(("Fuzzy match rate >= 70%", fm_rate >= 0.70, f"{fm_rate:.1%}"))
        checks.append(("Tamil-dominant >= 90%", tamil_rate >= 0.90, f"{tamil_rate:.1%}"))
        checks.append(("Repetition rate <= 10%", rep_rate <= 0.10, f"{rep_rate:.1%}"))

    if eval_results:
        em_rate = eval_summary["exact_match_rate"]
        fm_rate = eval_summary["fuzzy_match_rate"]
        tamil_rate = eval_summary["tamil_rate"]

        checks.append(("Eval exact match >= 30%", em_rate >= 0.30, f"{em_rate:.1%}"))
        checks.append(("Eval fuzzy match >= 60%", fm_rate >= 0.60, f"{fm_rate:.1%}"))
        checks.append(("Eval Tamil-dominant >= 90%", tamil_rate >= 0.90, f"{tamil_rate:.1%}"))

    for label, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        w(f"  [{status}] {label}: {value}")

    all_passed = all(p for _, p, _ in checks)
    w("")
    w(f"  Verdict: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    w("=" * 70)

    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")

    return report_text


def save_detailed_results(test_results, eval_results, output_dir):
    """Save detailed per-example results as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_results:
        path = output_dir / "test_case_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        print(f"Test case results: {path}")

    if eval_results:
        path = output_dir / "eval_set_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"Eval set results: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Tamil grammar correction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a HuggingFace model on test cases
  python evaluate_tamil_grammar.py --model Qwen/Qwen3-8B --mode test-cases

  # Test on eval.jsonl
  python evaluate_tamil_grammar.py --model Qwen/Qwen3-8B --mode eval-set --eval-file eval.jsonl

  # Test an Ollama model
  python evaluate_tamil_grammar.py --ollama tamil-llama --mode test-cases

  # Full evaluation
  python evaluate_tamil_grammar.py --model ./merged_model --mode all --eval-file eval.jsonl

  # Quick test (first 20 eval examples)
  python evaluate_tamil_grammar.py --model Qwen/Qwen3-8B --mode eval-set --eval-file eval.jsonl --max-samples 20
        """,
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        help="HuggingFace model name or local path",
    )
    model_group.add_argument(
        "--ollama",
        help="Ollama model name",
    )

    # Evaluation mode
    parser.add_argument(
        "--mode",
        choices=["test-cases", "eval-set", "all"],
        default="test-cases",
        help="Evaluation mode (default: test-cases)",
    )

    # Files
    parser.add_argument(
        "--eval-file",
        help="Path to eval.jsonl for eval-set mode",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory for output files (default: eval_results)",
    )
    parser.add_argument(
        "--report",
        default="eval_report.txt",
        help="Path to evaluation report (default: eval_report.txt)",
    )

    # Options
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max eval examples to test (for speed)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--compare-with",
        help="Second model to compare against (HuggingFace name or Ollama model)",
    )

    args = parser.parse_args()

    # Validate
    if args.mode in ["eval-set", "all"] and not args.eval_file:
        parser.error("--eval-file is required for eval-set and all modes")

    eval_file = Path(args.eval_file) if args.eval_file else None
    if eval_file and not eval_file.exists():
        parser.error(f"Eval file not found: {eval_file}")

    # Set up generator
    print("=" * 60)
    print("TAMIL GRAMMAR EVALUATION")
    print("=" * 60)

    if args.ollama:
        print(f"Model: Ollama/{args.ollama}")
        def generator_fn(prompt):
            return generate_ollama(args.ollama, prompt, SYSTEM_PROMPT, max_new_tokens=args.max_new_tokens)
        model_name = args.ollama
    else:
        print(f"Model: {args.model}")
        model, tokenizer = load_hf_model(args.model)
        def generator_fn(prompt):
            return generate_hf(model, tokenizer, prompt, SYSTEM_PROMPT, max_new_tokens=args.max_new_tokens)
        model_name = args.model

    print(f"Mode: {args.mode}")
    print()

    # Run evaluation
    test_results = []
    eval_results = []

    if args.mode in ["test-cases", "all"]:
        print("Evaluating on test cases...")
        test_results = evaluate_test_cases(generator_fn)
        print(f"  Completed {len(test_results)} test cases")

    if args.mode in ["eval-set", "all"]:
        print(f"Evaluating on eval set ({eval_file})...")
        eval_results = evaluate_eval_set(generator_fn, eval_file, args.max_samples)
        print(f"  Completed {len(eval_results)} examples")

    # Compute summaries
    test_summary = compute_summary(test_results) if test_results else {}
    eval_summary = compute_summary(eval_results) if eval_results else {}
    cat_summaries = compute_category_summary(test_results) if test_results else {}

    # Print quick summary
    print()
    print("=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    if test_results:
        print(f"  Test cases:  EM={test_summary['exact_match_rate']:.1%}  "
              f"FM={test_summary['fuzzy_match_rate']:.1%}  "
              f"Tamil={test_summary['tamil_rate']:.1%}")
    if eval_results:
        print(f"  Eval set:    EM={eval_summary['exact_match_rate']:.1%}  "
              f"FM={eval_summary['fuzzy_match_rate']:.1%}  "
              f"Tamil={eval_summary['tamil_rate']:.1%}")
    print("=" * 60)

    # Generate report
    report = generate_report(
        test_results, eval_results, test_summary, eval_summary,
        cat_summaries, args.report,
    )

    # Save detailed results
    save_detailed_results(test_results, eval_results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
