#!/usr/bin/env python3
"""
Tamil Grammar Dataset Cleaner & Validator
==========================================

Validates and cleans a Tamil grammar correction JSONL dataset.
Checks structure, Tamil content, duplicates, output format, length,
repetition, and generates a detailed quality report.

Usage:
    python clean_dataset.py input.jsonl
    python clean_dataset.py input.jsonl --output cleaned.jsonl --report report.txt
    python clean_dataset.py input.jsonl --fix --output cleaned.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from difflib import SequenceMatcher


# ============================================================
# Tamil Unicode range
# ============================================================
TAMIL_RANGE = (0x0B80, 0x0BFF)


def is_tamil_char(ch):
    """Check if a character is in the Tamil Unicode block."""
    cp = ord(ch)
    return TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1]


def tamil_ratio(text):
    """Return the fraction of non-space characters that are Tamil."""
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    tamil_count = sum(1 for c in chars if is_tamil_char(c))
    return tamil_count / len(chars)


def tamil_char_count(text):
    """Count Tamil characters in text."""
    return sum(1 for c in text if is_tamil_char(c))


def is_mostly_tamil(text, threshold=0.3):
    """Check if text is predominantly Tamil."""
    return tamil_ratio(text) >= threshold


# ============================================================
# Validation checks
# ============================================================

REQUIRED_FIELDS = ["instruction", "input", "output"]
OPTIONAL_FIELDS = ["correction", "error_type", "explanation"]
ALL_KNOWN_FIELDS = set(REQUIRED_FIELDS + OPTIONAL_FIELDS + ["id", "language", "task"])


def check_structure(item, line_num):
    """Validate JSON structure and required fields. Returns list of issues."""
    issues = []

    if not isinstance(item, dict):
        issues.append(f"Line {line_num}: Not a JSON object")
        return issues

    for field in REQUIRED_FIELDS:
        if field not in item:
            issues.append(f"Line {line_num}: Missing required field '{field}'")
        elif not isinstance(item[field], str):
            issues.append(f"Line {line_num}: Field '{field}' is not a string")
        elif not item[field].strip():
            issues.append(f"Line {line_num}: Field '{field}' is empty")

    # Warn about unexpected fields
    unknown = set(item.keys()) - ALL_KNOWN_FIELDS
    if unknown:
        issues.append(f"Line {line_num}: Unknown fields: {unknown}")

    return issues


def check_tamil_content(item, line_num):
    """Validate Tamil character presence. Returns list of issues."""
    issues = []
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    if not is_mostly_tamil(input_text, threshold=0.2):
        ratio = tamil_ratio(input_text)
        issues.append(
            f"Line {line_num}: Input has low Tamil ratio ({ratio:.1%}): "
            f"'{input_text[:80]}...'"
        )

    if not is_mostly_tamil(output_text, threshold=0.2):
        ratio = tamil_ratio(output_text)
        issues.append(
            f"Line {line_num}: Output has low Tamil ratio ({ratio:.1%}): "
            f"'{output_text[:80]}...'"
        )

    if tamil_char_count(input_text) < 5:
        issues.append(
            f"Line {line_num}: Input has very few Tamil chars "
            f"({tamil_char_count(input_text)}): '{input_text[:60]}'"
        )

    if tamil_char_count(output_text) < 10:
        issues.append(
            f"Line {line_num}: Output has very few Tamil chars "
            f"({tamil_char_count(output_text)}): '{output_text[:60]}'"
        )

    return issues


def check_lengths(item, line_num, max_input=500, max_output=2000):
    """Check field lengths. Returns list of issues."""
    issues = []
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    if len(input_text) > max_input:
        issues.append(
            f"Line {line_num}: Input too long ({len(input_text)} chars > {max_input})"
        )

    if len(output_text) > max_output:
        issues.append(
            f"Line {line_num}: Output too long ({len(output_text)} chars > {max_output})"
        )

    if len(input_text.split()) < 3:
        issues.append(
            f"Line {line_num}: Input too short ({len(input_text.split())} words)"
        )

    return issues


def check_duplicates(items):
    """Find exact and near-duplicate pairs. Returns dict of issues by line."""
    issues = {}

    # Exact duplicates
    seen_exact = {}
    for i, item in enumerate(items):
        key = (item.get("input", "").strip(), item.get("output", "").strip())
        if key in seen_exact:
            prev = seen_exact[key]
            issues.setdefault(i + 1, []).append(
                f"Exact duplicate of line {prev + 1}"
            )
        else:
            seen_exact[key] = i

    # Near-duplicates (input similarity > 0.85)
    inputs = [(i, item.get("input", "").strip()) for i, item in enumerate(items)]
    seen_fuzzy = []
    for i, text_i in inputs:
        for prev_i, text_j in seen_fuzzy:
            ratio = SequenceMatcher(None, text_i, text_j).ratio()
            if ratio > 0.85:
                issues.setdefault(i + 1, []).append(
                    f"Near-duplicate (sim={ratio:.2f}) of line {prev_i + 1}"
                )
                break
        seen_fuzzy.append((i, text_i))

    return issues


def check_output_format(item, line_num):
    """Validate output contains expected components. Returns list of issues."""
    issues = []
    output = item.get("output", "")

    # Check for correction marker
    has_correction = any(
        marker in output
        for marker in [
            "சரியான வாக்கியம்",
            "corrected",
            "Correction",
            "திருத்தம்",
            "பிழை",
        ]
    )
    if not has_correction:
        issues.append(f"Line {line_num}: Output missing correction marker")

    # Check for explanation
    has_explanation = any(
        marker in output
        for marker in [
            "விளக்கம்",
            "Explanation",
            "explanation",
            "காரணம்",
            "ஏன்",
        ]
    )
    if not has_explanation:
        issues.append(f"Line {line_num}: Output missing explanation")

    # Check input != output (ignoring the correction prefix)
    input_text = item.get("input", "").strip()
    # Strip common prefixes from output to compare core content
    output_core = output
    for prefix in ["சரியான வாக்கியம்:", "சரியான வாக்கியம் :", "Corrected:"]:
        if output_core.startswith(prefix):
            output_core = output_core[len(prefix) :].strip()
            break

    if input_text and output_core and input_text == output_core:
        issues.append(f"Line {line_num}: Input and output are identical")

    return issues


def check_repetition(item, line_num):
    """Detect repetitive content within a single example. Returns list of issues."""
    issues = []
    output = item.get("output", "")

    # Check for repeated phrases (3+ words repeated)
    words = output.split()
    if len(words) > 10:
        # Check for repeated 3-gram
        for n in [3, 4, 5]:
            ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            for ngram, count in ngram_counts.items():
                if count >= 3 and n >= 3:
                    phrase = " ".join(ngram)
                    issues.append(
                        f"Line {line_num}: Phrase repeated {count}x in output: '{phrase}'"
                    )
                    break  # Only report once per example

    # Check for repeated lines within output
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    if len(lines) > 1:
        line_counts = Counter(lines)
        for line, count in line_counts.items():
            if count >= 2 and len(line) > 10:
                issues.append(
                    f"Line {line_num}: Output line repeated {count}x: '{line[:60]}'"
                )

    return issues


def check_encoding(item, line_num):
    """Check for encoding issues. Returns list of issues."""
    issues = []

    for field in ["input", "output"]:
        text = item.get(field, "")
        # Check for mojibake patterns (common UTF-8 decoded as Latin-1)
        mojibake_patterns = [
            "\ufffd",  # replacement character
            "\xc3\xa0",  # common mojibake for à
        ]
        for pattern in mojibake_patterns:
            if pattern in text:
                issues.append(f"Line {line_num}: Possible encoding issue in {field}")

        # Check for mixed scripts that shouldn't be together
        has_tamil = any(is_tamil_char(c) for c in text)
        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
        if has_tamil and has_devanagari:
            issues.append(
                f"Line {line_num}: Mixed Tamil and Devanagari in {field}"
            )

    return issues


# ============================================================
# Cleaning / fixing
# ============================================================

def clean_output_text(text):
    """Clean common issues in output text."""
    # Remove leading/trailing whitespace
    text = text.strip()

    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing whitespace on each line
    lines = text.split("\n")
    lines = [l.rstrip() for l in lines]
    text = "\n".join(lines)

    return text


def fix_item(item):
    """Apply common fixes to an item. Returns fixed item or None if unfixable."""
    fixed = dict(item)

    # Clean output
    if "output" in fixed:
        fixed["output"] = clean_output_text(fixed["output"])

    # Clean input
    if "input" in fixed:
        fixed["input"] = fixed["input"].strip()

    # Clean instruction
    if "instruction" in fixed:
        fixed["instruction"] = fixed["instruction"].strip()

    # Remove empty output
    if not fixed.get("output", "").strip():
        return None

    # Remove empty input
    if not fixed.get("input", "").strip():
        return None

    return fixed


# ============================================================
# Report generation
# ============================================================

def generate_report(
    items,
    all_issues,
    line_issues,
    dup_issues,
    stats,
    output_path=None,
):
    """Generate a detailed quality report."""
    lines = []
    w = lines.append

    w("=" * 70)
    w("TAMIL GRAMMAR DATASET QUALITY REPORT")
    w("=" * 70)
    w("")

    # Summary stats
    w("--- SUMMARY ---")
    w(f"  Total examples:        {stats['total']}")
    w(f"  Valid examples:        {stats['valid']}")
    w(f"  Examples with issues:  {stats['with_issues']}")
    w(f"  Clean examples:        {stats['clean']}")
    w("")

    # Field statistics
    w("--- FIELD STATISTICS ---")
    for field in ["input", "output"]:
        lengths = [len(item.get(field, "")) for item in items]
        words = [len(item.get(field, "").split()) for item in items]
        tamil_r = [tamil_ratio(item.get(field, "")) for item in items]
        w(f"  {field}:")
        w(f"    Length (chars):  min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.0f}")
        w(f"    Length (words):  min={min(words)}, max={max(words)}, "
          f"avg={sum(words)/len(words):.1f}")
        w(f"    Tamil ratio:     min={min(tamil_r):.1%}, max={max(tamil_r):.1%}, "
          f"avg={sum(tamil_r)/len(tamil_r):.1%}")
    w("")

    # Issue breakdown
    w("--- ISSUE BREAKDOWN ---")
    issue_types = Counter()
    for issues in all_issues:
        for issue in issues:
            # Extract issue type (before the colon)
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_types[issue_type] += 1

    for issue_type, count in issue_types.most_common():
        try:
            w(f"  {issue_type}: {count}")
        except UnicodeEncodeError:
            safe = issue_type.encode('ascii', errors='replace').decode('ascii')
            w(f"  {safe}: {count}")
    w("")

    # Duplicate analysis
    w("--- DUPLICATE ANALYSIS ---")
    exact_dups = sum(1 for v in dup_issues.values()
                     if any("Exact duplicate" in i for i in v))
    near_dups = sum(1 for v in dup_issues.values()
                    if any("Near-duplicate" in i for i in v))
    w(f"  Exact duplicates:     {exact_dups}")
    w(f"  Near-duplicates:      {near_dups}")
    w("")

    # Tamil content analysis
    w("--- TAMIL CONTENT ANALYSIS ---")
    input_tamil_ratios = [tamil_ratio(item.get("input", "")) for item in items]
    output_tamil_ratios = [tamil_ratio(item.get("output", "")) for item in items]

    low_tamil_input = sum(1 for r in input_tamil_ratios if r < 0.3)
    low_tamil_output = sum(1 for r in output_tamil_ratios if r < 0.3)
    w(f"  Input  Tamil ratio < 30%:  {low_tamil_input} ({low_tamil_input/stats['total']:.1%})")
    w(f"  Output Tamil ratio < 30%:  {low_tamil_output} ({low_tamil_output/stats['total']:.1%})")
    w("")

    # Category distribution (from error_type field if present)
    if any("error_type" in item for item in items):
        w("--- ERROR TYPE DISTRIBUTION ---")
        error_types = Counter(
            item.get("error_type", "unknown") for item in items
        )
        for etype, count in error_types.most_common():
            try:
                w(f"  {etype}: {count}")
            except UnicodeEncodeError:
                safe = etype.encode('ascii', errors='replace').decode('ascii')
                w(f"  {safe}: {count}")
        w("")

    # Top worst examples
    w("--- TOP 10 WORST EXAMPLES (most issues) ---")
    worst = sorted(line_issues.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for line_num, issues in worst:
        w(f"  Line {line_num} ({len(issues)} issues):")
        for issue in issues[:3]:  # Show top 3 issues
            w(f"    - {issue}")
    w("")

    w("=" * 70)
    w("END OF REPORT")
    w("=" * 70)

    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")

    return report_text


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean and validate Tamil grammar correction dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_dataset.py train.jsonl
  python clean_dataset.py train.jsonl --output cleaned.jsonl --report report.txt
  python clean_dataset.py train.jsonl --fix --output cleaned.jsonl --verbose
  python clean_dataset.py train.jsonl --check-duplicates --max-dup-similarity 0.9
        """,
    )
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument(
        "-o", "--output",
        help="Path to write cleaned JSONL (only valid examples)",
    )
    parser.add_argument(
        "-r", "--report",
        default="cleaning_report.txt",
        help="Path to write quality report (default: cleaning_report.txt)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply automatic fixes before validation",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Enable duplicate checking (slow for large datasets)",
    )
    parser.add_argument(
        "--max-dup-similarity",
        type=float,
        default=0.85,
        help="Similarity threshold for near-duplicate detection (default: 0.85)",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=500,
        help="Maximum input field length in characters (default: 500)",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=2000,
        help="Maximum output field length in characters (default: 2000)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all issues to stdout",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print summary, no per-line issues",
    )

    args = parser.parse_args()

    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {input_path}")
    items = []
    parse_errors = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                items.append(item)
            except json.JSONDecodeError as e:
                parse_errors.append((line_num, str(e)))

    print(f"Loaded: {len(items)} examples")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)}")
        for line_num, err in parse_errors[:5]:
            print(f"  Line {line_num}: {err}")

    if not items:
        print("No valid items to process.")
        sys.exit(1)

    # Apply fixes if requested
    if args.fix:
        print("\nApplying fixes...")
        original_count = len(items)
        fixed_items = []
        for item in items:
            fixed = fix_item(item)
            if fixed is not None:
                fixed_items.append(fixed)
        items = fixed_items
        print(f"  Before: {original_count}")
        print(f"  After:  {len(items)}")
        print(f"  Removed: {original_count - len(items)}")

    # Run validation checks
    print("\nRunning validation checks...")
    all_issues = []
    line_issues = {}

    for i, item in enumerate(items):
        line_num = i + 1
        issues = []

        issues.extend(check_structure(item, line_num))
        issues.extend(check_tamil_content(item, line_num))
        issues.extend(
            check_lengths(item, line_num, args.max_input_length, args.max_output_length)
        )
        issues.extend(check_output_format(item, line_num))
        issues.extend(check_repetition(item, line_num))
        issues.extend(check_encoding(item, line_num))

        all_issues.append(issues)
        if issues:
            line_issues[line_num] = issues

    # Duplicate checking
    dup_issues = {}
    if args.check_duplicates:
        print("Checking duplicates (this may take a while)...")
        dup_issues = check_duplicates(items)
        for line_num, dups in dup_issues.items():
            all_issues[line_num - 1].extend(dups)
            line_issues.setdefault(line_num, []).extend(dups)

    # Statistics
    total = len(items)
    with_issues = len([i for i in all_issues if i])
    clean = total - with_issues

    stats = {
        "total": total,
        "valid": total,
        "with_issues": with_issues,
        "clean": clean,
    }

    # Print summary
    print(f"\nResults:")
    print(f"  Total examples:       {total}")
    print(f"  With issues:          {with_issues} ({with_issues/total:.1%})")
    print(f"  Clean (no issues):    {clean} ({clean/total:.1%})")

    if line_issues and not args.quiet:
        print(f"\nIssue details (showing first 20):")
        shown = 0
        for line_num in sorted(line_issues.keys()):
            if shown >= 20:
                remaining = len(line_issues) - shown
                print(f"  ... and {remaining} more examples with issues")
                break
            issues = line_issues[line_num]
            print(f"\n  Line {line_num} ({len(issues)} issues):")
            for issue in issues[:5]:
                try:
                    print(f"    - {issue}")
                except UnicodeEncodeError:
                    # Handle Tamil characters on Windows console
                    safe = issue.encode('ascii', errors='replace').decode('ascii')
                    print(f"    - {safe}")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more")
            shown += 1

    # Generate report
    report_path = Path(args.report)
    report = generate_report(items, all_issues, line_issues, dup_issues, stats, report_path)
    if not args.quiet:
        print(f"\nFull report: {report_path}")

    # Write cleaned output
    if args.output:
        output_path = Path(args.output)
        # Filter out items with critical issues (missing fields, no Tamil)
        critical_fields = {"Missing required field", "Input has very few Tamil chars"}
        cleaned_count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for i, (item, issues) in enumerate(zip(items, all_issues)):
                has_critical = False
                for issue in issues:
                    for critical in critical_fields:
                        if critical in issue:
                            has_critical = True
                            break
                if not has_critical:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    cleaned_count += 1

        print(f"\nCleaned dataset written: {output_path}")
        print(f"  Kept:    {cleaned_count} examples")
        print(f"  Removed: {total - cleaned_count} examples")

    print("\nDone!")


if __name__ == "__main__":
    main()
