#!/usr/bin/env python3
"""
Tamil Grammar Correction - Fine-Tuning Pipeline
=================================================

Complete pipeline: data generation -> cleaning -> QLoRA training -> evaluation.
Optimized for RTX 3050 6GB / Colab T4.

Usage:
    # Step 1: Generate training data
    python finetune_tamil_grammar.py --step generate --samples 500

    # Step 2: Train the model
    python finetune_tamil_grammar.py --step train --data train.jsonl

    # Step 3: Evaluate
    python finetune_tamil_grammar.py --step evaluate --model ./output/merged

    # Run everything
    python finetune_tamil_grammar.py --step all --samples 500
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    # Model
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "use_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj",
    ],

    # Training
    "num_epochs": 3,
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 512,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 3,
    "seed": 42,

    # Data generation
    "generation_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gen_batch_size": 5,
    "gen_max_tokens": 300,
}

TAMIL_RANGE = (0x0B80, 0x0BFF)

GRAMMAR_CATEGORIES = [
    "எழுத்துப் பிழை",
    "சொல் பிழை",
    "வினைச்சொல் பயன்பாட்டுப் பிழை",
    "காலப் பிழை",
    "ஒருமை பன்மை பிழை",
    "வேற்றுமை உருபுப் பிழை",
    "இடைச்சொல் பயன்பாட்டுப் பிழை",
    "பெயர்ச்சொல் பயன்பாட்டுப் பிழை",
    "வாக்கிய அமைப்புப் பிழை",
    "சொல் வரிசைப் பிழை",
    "சந்திப் பிழை",
    "உருபுப் பிழை",
    "பால் மற்றும் எண்ணிக்கைப் பிழை",
    "காலம் மற்றும் வினைச்சொல் பிழை",
    "எழுத்து மற்றும் உச்சரிப்புப் பிழை",
    "பேச்சுத்தமிழ் மற்றும் எழுத்துத்தமிழ் வேறுபாடு",
    "மரியாதை மொழிப் பிழை",
    "எதிர்மறை வாக்கியப் பிழை",
    "கேள்வி வாக்கிய அமைப்புப் பிழை",
    "கலப்பு இலக்கணப் பிழை",
]

# ============================================================
# Step 1: Data Generation
# ============================================================

DATA_GEN_PROMPT = """Generate {count} Tamil grammar correction examples for the category "{category}".

Return ONLY a JSON array. Each object must have:
- "input": A Tamil sentence with a grammar error
- "correction": The corrected Tamil sentence
- "explanation": Tamil explanation of the error

Rules:
- Use natural modern Tamil
- Each example must be different
- Minimum 4 Tamil words per input
- No English words unless part of the error
- Return JSON array only, no markdown

Output:"""


def is_tamil_char(ch):
    cp = ord(ch)
    return TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1]


def tamil_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if is_tamil_char(c)) / len(chars)


def extract_json_array(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        try:
            fixed = re.sub(r",\s*]", "]", text[start:end + 1])
            return json.loads(fixed)
        except Exception:
            return None


def generate_dataset(num_samples):
    """Generate Tamil grammar training data using a small model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = CONFIG["generation_model"]
    print(f"\n{'='*60}")
    print(f"STEP 1: Generating {num_samples} training examples")
    print(f"Using model: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading generation model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!\n")

    # Generate examples
    all_examples = []
    seen = set()
    attempts = 0
    max_attempts = num_samples * 5

    while len(all_examples) < num_samples and attempts < max_attempts:
        attempts += 1
        category = GRAMMAR_CATEGORIES[len(all_examples) % len(GRAMMAR_CATEGORIES)]
        batch_size = min(CONFIG["gen_batch_size"], num_samples - len(all_examples))

        prompt = DATA_GEN_PROMPT.format(count=batch_size, category=category)

        messages = [
            {"role": "system", "content": "You are a Tamil grammar expert. Output JSON only."},
            {"role": "user", "content": prompt},
        ]

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["gen_max_tokens"],
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        parsed = extract_json_array(response)
        if not parsed:
            continue

        for item in parsed:
            if not isinstance(item, dict):
                continue
            inp = item.get("input", "").strip()
            corr = item.get("correction", "").strip()
            expl = item.get("explanation", "").strip()

            if not inp or not corr:
                continue
            if tamil_ratio(inp) < 0.3 or tamil_ratio(corr) < 0.3:
                continue
            if len(inp.split()) < 3:
                continue
            if inp == corr:
                continue

            key = (inp.lower(), corr.lower())
            if key in seen:
                continue
            seen.add(key)

            # Format as SFT output
            output_text = (
                f"சரியான வாக்கியம்: {corr}\n\n"
                f"பிழை வகை: {category}\n\n"
                f"விளக்கம்: {expl}"
            )

            all_examples.append({
                "instruction": "இந்த தமிழ் வாக்கியத்தில் உள்ள இலக்கணப் பிழையை திருத்தவும்.",
                "input": inp,
                "output": output_text,
                "error_type": category,
            })

            if len(all_examples) % 50 == 0:
                print(f"  Generated: {len(all_examples)}/{num_samples}")

        # Free memory
        del outputs, inputs
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nGenerated {len(all_examples)} examples")

    # Shuffle and split
    random.seed(CONFIG["seed"])
    random.shuffle(all_examples)

    split = int(len(all_examples) * 0.95)
    train_data = all_examples[:split]
    eval_data = all_examples[split:]

    # Save
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    for name, data in [("train.jsonl", train_data), ("eval.jsonl", eval_data)]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {path}: {len(data)} examples")

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return str(output_dir / "train.jsonl"), str(output_dir / "eval.jsonl")


# ============================================================
# Step 2: Training
# ============================================================

def train_model(train_file, eval_file):
    """Fine-tune model with QLoRA."""
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"\n{'='*60}")
    print(f"STEP 2: QLoRA Fine-Tuning")
    print(f"Model: {CONFIG['model_name']}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading dataset...")
    raw_train = load_dataset("json", data_files=train_file, split="train")
    raw_eval = load_dataset("json", data_files=eval_file, split="train")
    print(f"  Train: {len(raw_train)} examples")
    print(f"  Eval:  {len(raw_eval)} examples")

    # Format for SFT
    def format_example(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        user_msg = f"{instruction}\n\n{input_text}" if input_text else instruction
        return {"messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output_text},
        ]}

    train_dataset = raw_train.map(format_example, remove_columns=raw_train.column_names)
    eval_dataset = raw_eval.map(format_example, remove_columns=raw_eval.column_names)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in 4-bit
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["use_4bit"],
        bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, CONFIG["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=CONFIG["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["per_device_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        optim=CONFIG["optim"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        max_grad_norm=CONFIG["max_grad_norm"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_ratio=CONFIG["warmup_ratio"],
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        group_by_length=True,
        logging_steps=CONFIG["logging_steps"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=CONFIG["seed"],
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="messages",
        max_seq_length=CONFIG["max_seq_length"],
        packing=False,
    )

    # Train
    print(f"\nStarting training ({CONFIG['num_epochs']} epochs)...")
    print(f"  Effective batch: {CONFIG['per_device_batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"  LR: {CONFIG['learning_rate']}")
    print()

    train_result = trainer.train()

    # Save LoRA adapter
    lora_dir = output_dir / "lora_adapter"
    trainer.model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"\nLoRA adapter saved to: {lora_dir}")

    # Print metrics
    metrics = train_result.metrics
    print(f"\nTraining metrics:")
    print(f"  Loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Runtime: {metrics.get('train_runtime', 0):.0f}s")

    # Merge and save
    print("\nMerging LoRA into base model...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(lora_dir))
    model = model.merge_and_unload()

    merged_dir = output_dir / "merged"
    model.save_pretrained(str(merged_dir))
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(merged_dir))

    print(f"Merged model saved to: {merged_dir}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return str(merged_dir)


# ============================================================
# Step 3: Evaluation
# ============================================================

# 20 test cases (subset of the full 40)
TEST_CASES = [
    ("Tense", "நான் நேற்று பள்ளிக்கு போகிறேன்.", "நான் நேற்று பள்ளிக்குப் போனேன்."),
    ("Tense", "அவள் நேற்று சந்தைக்கு செல்கிறாள்.", "அவள் நேற்று சந்தைக்குச் சென்றாள்."),
    ("Agreement", "நான் தமிழ் பேசுகிறது.", "நான் தமிழ் பேசுகிறேன்."),
    ("Agreement", "அவள் நல்ல மாணவன்.", "அவள் நல்ல மாணவி."),
    ("Gender", "இந்த பெண் ஒரு நல்ல மாணவன்.", "இந்த பெண் ஒரு நல்ல மாணவி."),
    ("Gender", "அவள் நல்ல தலைவன்.", "அவள் நல்ல தலைவர்."),
    ("Morphology", "அவன் அழுகிறது.", "அவன் அழுகிறான்."),
    ("Morphology", "நான் சாப்பிட்டுக் கொண்டிருந்தான்.", "நான் சாப்பிட்டுக் கொண்டிருந்தேன்."),
    ("Word Order", "சமையல் அவன் செய்கிறான்.", "அவன் சமையல் செய்கிறான்."),
    ("Register", "அவங்க எப்ப வர்றாங்க.", "அவர்கள் எப்போது வருவார்கள்."),
    ("Register", "உங்களுக்கு என்ன ப்ராப்லம்?", "உங்களுக்கு என்ன பிரச்சனை?"),
    ("Code-mixed", "அவள் computer-ல் வேலை செய்கிறாள்.", "அவள் கணினியில் வேலை செய்கிறாள்."),
    ("Code-mixed", "நான் library-க்கு போனேன்.", "நான் நூலகத்திற்கு போனேன்."),
    ("Punctuation", "நான் வருகிறேன் நான்.", "நான் வருகிறேன்."),
    ("Ambiguous", "வானம் உயரம்.", "வானம் உயரமானது."),
    ("Case Markers", "அவன் வீட்டில் போனான்.", "அவன் வீட்டிற்கு சென்றான்."),
    ("Case Markers", "நான் நடக்கிறேன் நூலகம்.", "நான் நூலகத்திற்கு நடக்கிறேன்."),
    ("Tense", "நான் கடந்த மாதம் திரைப்படம் பார்க்கிறேன்.", "நான் கடந்த மாதம் திரைப்படம் பார்த்தேன்."),
    ("Register", "நாங்க இல்ல போய் வந்தோம்.", "நாம் இல்லத்துக்கு போய் வந்தோம்."),
    ("Ambiguous", "நண்பன் அவன்.", "அவன் நண்பன்."),
]


def normalize_tamil(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,!?;:\u0BCD\u0BD7\u0BF4]", "", text)
    return text.lower().strip()


def evaluate_model(model_path):
    """Evaluate the fine-tuned model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"STEP 3: Evaluation")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!\n")

    system_prompt = (
        "You are a Tamil grammar assistant. Correct grammar errors in Tamil sentences. "
        "Output only the corrected sentence."
    )

    # Run test cases
    results = []
    exact_matches = 0
    fuzzy_matches = 0

    print("Running test cases...")
    for i, (category, wrong, expected) in enumerate(TEST_CASES):
        prompt = f"திருத்தப்பட்ட உரை:\n{wrong}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        # Check match
        norm_pred = normalize_tamil(response)
        norm_exp = normalize_tamil(expected)
        em = norm_pred == norm_exp

        from difflib import SequenceMatcher
        fm_ratio = SequenceMatcher(None, norm_pred, norm_exp).ratio()
        fm = fm_ratio >= 0.8

        if em:
            exact_matches += 1
        if fm:
            fuzzy_matches += 1

        status = "PASS" if em else ("FUZZY" if fm else "FAIL")
        results.append({
            "category": category,
            "input": wrong,
            "expected": expected,
            "predicted": response,
            "exact_match": em,
            "fuzzy_ratio": fm_ratio,
        })

        try:
            print(f"  [{status}] {category}: {wrong[:30]}...")
        except UnicodeEncodeError:
            print(f"  [{status}] {category}: (Tamil text)")

    # Summary
    n = len(TEST_CASES)
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Exact matches:  {exact_matches}/{n} ({exact_matches/n:.1%})")
    print(f"  Fuzzy matches:  {fuzzy_matches}/{n} ({fuzzy_matches/n:.1%})")

    # Per-category
    cats = Counter(r["category"] for r in results)
    cat_em = Counter(r["category"] for r in results if r["exact_match"])
    print(f"\n  Per-category:")
    for cat in sorted(cats.keys()):
        em = cat_em.get(cat, 0)
        total = cats[cat]
        print(f"    {cat:<20s}: {em}/{total}")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {results_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Tamil Grammar Fine-Tuning Pipeline")
    parser.add_argument(
        "--step",
        choices=["generate", "train", "evaluate", "all"],
        required=True,
        help="Pipeline step to run",
    )
    parser.add_argument("--samples", type=int, default=500, help="Number of training samples to generate")
    parser.add_argument("--data", type=str, default=None, help="Path to train.jsonl")
    parser.add_argument("--eval-data", type=str, default=None, help="Path to eval.jsonl")
    parser.add_argument("--model", type=str, default=None, help="Path to merged model for evaluation")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")

    args = parser.parse_args()

    # Apply overrides
    if args.epochs:
        CONFIG["num_epochs"] = args.epochs
    if args.lr:
        CONFIG["learning_rate"] = args.lr
    if args.batch_size:
        CONFIG["per_device_batch_size"] = args.batch_size

    # Print config
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()

    if args.step in ["generate", "all"]:
        train_file, eval_file = generate_dataset(args.samples)
    else:
        train_file = args.data
        eval_file = args.eval_data

    if args.step in ["train", "all"]:
        if not train_file or not Path(train_file).exists():
            print(f"ERROR: train file not found: {train_file}")
            sys.exit(1)
        if not eval_file or not Path(eval_file).exists():
            print(f"ERROR: eval file not found: {eval_file}")
            sys.exit(1)
        merged_path = train_model(train_file, eval_file)
    else:
        merged_path = args.model

    if args.step in ["evaluate", "all"]:
        if not merged_path or not Path(merged_path).exists():
            print(f"ERROR: model not found: {merged_path}")
            sys.exit(1)
        evaluate_model(merged_path)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
