# Tamil-LLaMA Project Documentation

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Gradio Web App](#gradio-web-app)
  - [CLI Test Runner](#cli-test-runner)
- [Configuration](#configuration)
  - [Ollama Modelfile](#ollama-modelfile)
  - [LM Studio Config](#lm-studio-config)
- [Training Pipeline](#training-pipeline)
  - [SentencePiece Tokenizer](#sentencepiece-tokenizer)
  - [Pretraining](#pretraining)
  - [Fine-tuning](#fine-tuning)
  - [Push to Hub / Merge Adapter](#push-to-hub--merge-adapter)
- [Evaluation](#evaluation)
  - [Evaluation Criteria & Metrics](#evaluation-criteria--metrics)
  - [40-Case Test Suite](#40-case-test-suite)
- [Prompt Engineering](#prompt-engineering)
  - [System Prompt Design](#system-prompt-design)
  - [Safety & Fallback](#safety--fallback)
  - [Model Parameters](#model-parameters)
- [File Structure](#file-structure)
- [Grammar Correction Rules](#grammar-correction-rules)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)

---

## Project Overview

Tamil-LLaMA is a family of LLaMA-based language models focused on the Tamil language. It builds upon Meta's LLaMA 2, extending it with additional Tamil tokens and LoRA-based fine-tuning. This repository includes:

- A **Gradio web app** for real-time Tamil text grammar correction
- **CLI test scripts** to benchmark correction accuracy
- **Training scripts** for tokenizer training, pretraining, and fine-tuning
- **Evaluation scripts** using ChatGPT as a judge

The application uses **Ollama** to run the `tamil-llama` model locally, with no Hugging Face API dependency at runtime.

**Technical Report:** [https://arxiv.org/abs/2311.05845](https://arxiv.org/abs/2311.05845)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Gradio UI (app.py)                 │
│              http://localhost:7860                    │
│                                                       │
│  ┌─────────────┐         ┌──────────────────┐        │
│  │  Input Text  │  ───►  │  Ollama API       │       │
│  │  (Tamil)     │         │  POST /generate   │       │
│  └─────────────┘         │  model:           │       │
│                           │  tamil-llama      │       │
│  ┌─────────────┐         │                   │       │
│  │ Output Text  │  ◄───  │  Response (fixed) │       │
│  │ (Corrected)  │         └────────┬─────────┘       │
│  └─────────────┘                   │                  │
│                                    │                  │
└────────────────────────────────────┼──────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Ollama Runtime     │
                          │   localhost:11434    │
                          │                      │
                          │   tamil-llama:latest  │
                          │   (2.0 GB GGUF)      │
                          │   Q4_K_M quantized   │
                          └──────────────────────┘
```

**Key components:**

| Component | Description |
|-----------|-------------|
| `app.py` | Gradio web interface for Tamil text correction |
| `test_cases.py` | CLI test runner with basic system prompt |
| `test_cases_v2.py` | CLI test runner with improved grammar-aware system prompt |
| `config/ollama/Modelfile` | Ollama model definition (template, parameters, system prompt) |
| `config/lm_studio/model_config.json` | LM Studio preset configuration |
| `scripts/train/` | Training pipeline (tokenizer, pretrain, finetune) |
| `scripts/eval/` | Evaluation pipeline using ChatGPT-as-judge |

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| [Ollama](https://ollama.com/) | Latest | Runs the Tamil-LLaMA model locally |
| Python | 3.10+ | Application runtime |
| `gradio` | 6.x | Web UI framework |
| `requests` | 2.x | HTTP client for Ollama API |

**Optional (for training/evaluation):**
- `transformers`, `peft`, `trl`, `datasets` — for training scripts
- `sentencepiece` — for tokenizer training
- OpenAI API key — for evaluation with ChatGPT

---

## Installation & Setup

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com/). Verify installation:

```bash
ollama --version
```

### 2. Create the Tamil-LLaMA Model in Ollama

Place the GGUF model file and Modelfile in the same directory, then:

```bash
cd config/ollama
ollama create tamil-llama -f Modelfile
```

Verify the model is loaded:

```bash
ollama list
```

You should see:

```
NAME                  ID              SIZE      MODIFIED
tamil-llama:latest    <hash>         2.0 GB    ...
```

### 3. Install Python Dependencies

The app requires only two packages:

```bash
pip install gradio requests
```

### 4. Run the Application

```bash
# Start Ollama (if not already running)
ollama serve

# Launch the Gradio app
python app.py
```

The app will be available at **http://localhost:7860**.

---

## Usage

### Gradio Web App

**Starting the app:**

```bash
python app.py
```

**Using the app:**

1. Open http://localhost:7860 in your browser
2. Enter Tamil text with grammar errors in the **Input Text** field
3. Click **பிழையை திருத்து (Correct)** or press Enter
4. View the corrected Tamil text in the **Output Text** field

**Built-in examples** (expand the Examples section):

| Input (with errors) | Expected Output (corrected) |
|---------------------|----------------------------|
| நான் நேற்று பள்ளிக்கு போகிறேன். | நான் நேற்று பள்ளிக்குப் போனேன். |
| நான் தமிழ் பேசுகிறது. | நான் தமிழ் பேசுகிறேன். |
| அவள் நல்ல மாணவன். | அவள் நல்ல மாணவி. |
| எனக்கு ஒரு புத்தகம் இருக்கிறான். | என்னிடம் ஒரு புத்தகம் இருக்கிறது. |

**App configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `server_name` | `0.0.0.0` | Listen on all interfaces |
| `server_port` | `7860` | Web server port |
| `temperature` | `0.1` | Low temperature for deterministic corrections |
| `repeat_penalty` | `1.1` | Prevents repetitive output |
| `timeout` | `120s` | API request timeout |

### CLI Test Runner

**Basic test (test_cases.py):**

```bash
python test_cases.py
```

**Improved test with grammar-aware prompt (test_cases_v2.py):**

```bash
python test_cases_v2.py
```

Both scripts run 10 test cases and report pass/fail. A test passes if the expected corrected text appears in the model's response.

---

## Configuration

### Ollama Modelfile

Location: `config/ollama/Modelfile`

```dockerfile
FROM ./tamil-llama-v0.2-q4.gguf

TEMPLATE """{{- if .First }}{{ .System }}{{- end }}

### Instruction:
{{ .Prompt }}

### Response:
"""

SYSTEM """..."""  # Tamil safety/helper system prompt

PARAMETER temperature 0.3
PARAMETER repeat_penalty 1.2
PARAMETER num_predict -1
```

**Model parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `temperature` | 0.3 | Balanced creativity/accuracy |
| `repeat_penalty` | 1.2 | Reduces repetition |
| `num_predict` | -1 | No token limit on output |

**Optional parameters** (for resource-constrained systems):

```
PARAMETER num_thread 8    # CPU threads
PARAMETER num_gpu 0       # GPU layers (0 = CPU only)
```

### LM Studio Config

Location: `config/lm_studio/model_config.json`

Use this to import a preset in LM Studio's Chat tab:

1. Open LM Studio → Chat tab
2. Click Preset dropdown → "Import Preset From File"
3. Select `config/lm_studio/model_config.json`

Key settings:

| Setting | Value |
|---------|-------|
| `n_ctx` | 2048 (context window) |
| `n_gpu_layers` | 20 |
| `temp` | 0.6 |
| `top_k` | 40 |
| `top_p` | 0.95 |
| `repeat_penalty` | 1.1 |

---

## Training Pipeline

### SentencePiece Tokenizer

Train a custom SentencePiece tokenizer for Tamil text:

```bash
cd scripts/train/sentencepiece
python train.py
```

Key files:

| File | Purpose |
|------|---------|
| `train.py` | Trains the SentencePiece model |
| `test.py` | Tests the trained tokenizer |
| `merge_tokenizer.py` | Merges the custom tokenizer with LLaMA tokenizer |
| `generate_text_corpus.py` | Generates training corpus data |

### Pretraining

Continual pretraining on Tamil text data using LoRA/PEFT:

```bash
cd scripts/train/pretrain
bash run_pt.sh
```

Key files:

| File | Purpose |
|------|---------|
| `run_clm_with_peft.py` | Main pretraining script with PEFT/LoRA |
| `flash_attn_patch.py` | Flash Attention memory optimization |
| `run_pt.sh` | Shell script with default training arguments |

### Fine-tuning

Fine-tune on instruction data (Tamil Alpaca):

```bash
cd scripts/train/finetune
bash run_finetuning.sh
```

Key files:

| File | Purpose |
|------|---------|
| `finetune.py` | Main fine-tuning script |
| `make_shards.py` | Splits large datasets into manageable shards |
| `run_finetuning.sh` | Shell script with training arguments |

### Push to Hub / Merge Adapter

After training:

```bash
# Push LoRA adapter to Hugging Face Hub
python scripts/train/utils/push_to_hub.py

# Merge LoRA adapter with base model
python scripts/train/utils/merge_adapter.py
```

---

## Evaluation

### Evaluation Criteria & Metrics

| Criterion | Description | Metrics / Measurement |
|-----------|-------------|----------------------|
| **Accuracy** | Correctness of corrections (avoiding missed errors or wrong fixes). | Precision/F1 against gold-corrected corpus; F₀.5 for precision emphasis; GLEU (BLEU-like) for overlap with references. |
| **Fluency** | How natural and grammatically fluent the output is. | Perplexity (lower = better); Human rating on Likert scale; GLEU n-gram fluency. |
| **Conservativeness** | Tendency to leave correct text unchanged (avoid unnecessary edits). | No-Edit Rate: fraction of correct inputs returned unchanged. Should be ~100%. |
| **Hallucination Rate** | Rate of adding unrelated or false content. | Count outputs with new content not implied by input. Use LLM entailment checks or human audit. |
| **Explainability** | Quality of explanations (if asked). | Clarity/correctness of rationale; human scoring. |

Each metric should be computed over a held-out Tamil error-correction dataset (e.g. INDIC-GEC test or the 40-case test suite).

### Automated Evaluation with ChatGPT-as-Judge

```bash
cd scripts/eval
python run_eval.py --input-csv <predictions.csv> --output-csv <scores.csv>
```

This uses GPT-4 to score model outputs on a 1-10 scale.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-csv` | (required) | CSV with model predictions |
| `--output-csv` | `tamil_eval_scores.csv` | Output CSV with scores |
| `--model-output-field` | `tamil-llama` | Column name of model outputs |
| `--max-retries` | `3` | Max retries on rate limit |
| `--min-wait-time` | `30` | Seconds between retries |

### Local Test Cases

```bash
# Basic grammar correction tests (10 cases)
python test_cases.py

# Tests with grammar-aware prompting (10 cases)
python test_cases_v2.py

# Full 40-case test suite across 10 categories
python test_cases_v3.py
```

### 40-Case Test Suite

The comprehensive test suite (`test_cases_v3.py`) covers 40 Tamil sentences across 10 error categories:

| # | Category | Cases | What It Tests |
|---|----------|-------|---------------|
| 1 | **Tense** | 4 | Past/present/future verb forms (e.g. நேற்று requires past tense) |
| 2 | **Agreement** | 4 | Subject-verb person/number matching |
| 3 | **Gender** | 4 | Noun/adjective gender forms (மாணவன் → மாணவி for female) |
| 4 | **Case Markers** | 4 | Postposition suffixes (~இல் vs ~இற்கு) |
| 5 | **Word Order** | 4 | SOV order and clause structure |
| 6 | **Morphology** | 4 | Verb inflections, affixes, particles |
| 7 | **Register** | 4 | Colloquial → formal (அவங்க → அவர்கள்) |
| 8 | **Code-mixed** | 4 | English → Tamil (computer → கணினி) |
| 9 | **Punctuation** | 4 | Sentence splitting, emoji removal |
| 10 | **Ambiguous** | 4 | Short phrases needing completion |

**Running the full suite:**

```bash
python test_cases_v3.py
```

**Sample output:**

```
Test #01 [Tense] — PASS
  Input:    நான் நேற்று பள்ளிக்கு போகிறேன்.
  Expected: நான் நேற்று பள்ளிக்குப் போனேன்.
  Got:      நான் நேற்று பள்ளிக்குப் போனேன்.

RESULTS BY CATEGORY
  Agreement            4/4 passed  (100%)
  Tense                4/4 passed  (100%)
  ...
OVERALL: 35 passed, 5 failed out of 40
ACCURACY: 87.5%
```

---

## Prompt Engineering

### System Prompt Design

The system prompt follows best practices from prompt engineering research and Tamil GEC (Grammatical Error Correction) studies:

```
You are a Tamil grammar and writing assistant. Your task is to check Tamil
input sentences for errors and correct them. You should identify and fix all
grammatical, spelling, punctuation, and syntactic mistakes without changing
the original meaning or tone.

Output only the corrected Tamil sentence (no quotes or extra explanation).
If the input is already grammatically correct, return it verbatim (unchanged).
Do NOT hallucinate or add information not present in the input.
```

**Key design principles:**

| Principle | Implementation |
|-----------|---------------|
| **Role Definition** | Explicitly assigns the model as a "Tamil grammar and writing assistant" |
| **Scope** | Lists specific error categories (grammar, spelling, punctuation, syntax) and Tamil-specific concerns (verb conjugation, case suffixes, gender) |
| **Preserve Meaning** | Stresses retaining meaning, tone, and script to prevent content alteration |
| **Output Format** | Demands only the corrected sentence — no extra labels or explanation |
| **Idempotence** | "If correct, return unchanged" — avoids unnecessary edits |
| **No Hallucinations** | Explicitly forbids adding unrelated content |
| **Language Constraint** | Forbids translation; must stay in Tamil |
| **Safety Clause** | Provides a Tamil refusal for harmful input |
| **Example Behavior** | Concrete input→output example reinforces expected style |

### Safety & Fallback

The system includes a safety clause for disallowed content:

```python
# If harmful content is detected:
"மன்னிக்கவும், உதவி செய்ய முடியவில்லை."
```

In production, integrate a Tamil content classifier or keyword filter to catch taboo words before the model processes them.

### Model Parameters

The recommended Ollama parameters balance precision vs. creativity:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.2 | Low temperature for deterministic, rule-governed edits |
| `top_p` | 0.9 | Moderately restrictive sampling |
| `repeat_penalty` | 1.1 | Prevents repetitive output |
| `num_predict` | -1 | No token limit (generate until end-of-sequence) |

---

## File Structure

```
tamil-llama/
├── app.py                          # Gradio web application
├── test_cases.py                   # CLI test runner (10 cases)
├── test_cases_v2.py                # CLI test runner (10 cases, improved prompt)
├── test_cases_v3.py                # Full 40-case test suite (10 categories)
├── tamil-llama-q4.gguf            # GGUF model file (v0.1)
├── tamil-llama-v0.2-q4.gguf      # GGUF model file (v0.2)
├── requirements.txt                # Conda environment spec
├── DOCUMENTATION.md                # This file
├── README.md                       # Project readme
├── LICENSE                         # GNU GPL v3.0
├── LLAMA2-LICENSE                  # LLaMA 2 license
├── CITATION.cff                    # Citation metadata
│
├── config/
│   ├── ollama/
│   │   ├── Modelfile               # Ollama model definition
│   │   └── tamil-llama-v0.2-q4.gguf
│   └── lm_studio/
│       └── model_config.json       # LM Studio preset
│
├── scripts/
│   ├── train/
│   │   ├── sentencepiece/
│   │   │   ├── train.py            # Tokenizer training
│   │   │   ├── test.py             # Tokenizer testing
│   │   │   ├── merge_tokenizer.py  # Tokenizer merge
│   │   │   └── generate_text_corpus.py
│   │   ├── pretrain/
│   │   │   ├── run_clm_with_peft.py  # Pretraining script
│   │   │   ├── flash_attn_patch.py   # Flash Attention
│   │   │   ├── run_pt.sh             # Training launcher
│   │   │   └── README.md
│   │   ├── finetune/
│   │   │   ├── finetune.py           # Fine-tuning script
│   │   │   ├── make_shards.py        # Data sharding
│   │   │   └── run_finetuning.sh     # Training launcher
│   │   └── utils/
│   │       ├── push_to_hub.py        # Push to HF Hub
│   │       └── merge_adapter.py      # Merge LoRA adapter
│   ├── eval/
│   │   ├── run_eval.py             # ChatGPT-as-judge evaluation
│   │   └── chatgpt_preds.py        # Generate ChatGPT predictions
│   └── utils/
│       └── count_indic_tokens.py   # Token counting utility
│
├── assets/                         # Images and screenshots
└── __pycache__/                    # Python cache
```

---

## Grammar Correction Rules

The app enforces these Tamil grammar rules through the system prompt:

### 1. Person Agreement (கருத்துரு முரண்)

| Person | Tamil | Verb Ending |
|--------|-------|-------------|
| 1st person | நான் | கிறேன் |
| 2nd person (male) | நீ | கிறாய் |
| 3rd person (male) | அவன் | கிறான் |
| 3rd person (female) | அவள் | கிறாள் |
| Plural | அவர்கள் | கிறார்கள் |

### 2. Tense Agreement (கால முரண்)

- "நேற்று" (yesterday) requires past tense: போனேன், சென்றான், வந்தாள்
- "தினமும்" (daily) requires present tense: போகிறேன், செல்கிறான்

### 3. Gender Agreement (பால் முரண்)

| Gender | Pronoun | Noun |
|--------|---------|------|
| Male | அவன் | பையன் / மாணவன் |
| Female | அவள் | பெண் / மாணவி |

### 4. Possession Agreement (உரிமை முரண்)

- "எனக்கு" → "என்னிடம்" when referring to objects/possession

### 5. Sandhi Rules (இணைப்பு முரண்)

- Before words starting with வன்மை (hard consonants), use "ப்" link: பள்ளிக்கு**ப்** போனேன்

---

## Troubleshooting

### Port Already in Use

```
OSError: Cannot find empty port in range: 7860-7860
```

**Solution:** Kill the conflicting process:

```bash
# Find process on port 7860
netstat -ano | grep 7860 | grep LISTEN

# Kill it (replace <PID> with the actual PID)
taskkill /PID <PID> /F   # Windows
kill -9 <PID>             # Linux/Mac
```

### Ollama Connection Error

```
Ollama இயங்கவில்லை (Ollama is not running)
```

**Solution:** Start Ollama:

```bash
ollama serve
```

### Timeout Error

```
காலாவதியானது (Timed out)
```

**Solution:** The model may be loading for the first time. Wait and retry, or increase the timeout in `app.py`.

### Windows Console Encoding Error (Tamil Characters)

```
UnicodeEncodeError: 'charmap' codec can't encode characters
```

**Solution:** The app handles this internally via `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`. If running custom scripts, add this at the top.

### Model Not Found

```
Error: model "tamil-llama" not found
```

**Solution:** Create the model in Ollama:

```bash
cd config/ollama
ollama create tamil-llama -f Modelfile
```

---

## License

This project is licensed under **GNU GPL v3.0** — see [LICENSE](LICENSE).

> **Note:** As a derivative of Meta's LLaMA 2 model, it is also subject to the [LLAMA2-LICENSE](LLAMA2-LICENSE).

---

## Citation

```bibtex
@misc{balachandran2023tamilllama,
      title={Tamil-Llama: A New Tamil Language Model Based on Llama 2},
      author={Abhinand Balachandran},
      year={2023},
      eprint={2311.05845},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
