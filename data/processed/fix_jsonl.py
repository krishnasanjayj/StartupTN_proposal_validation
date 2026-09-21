"""
Auto-repair script for JSONL files where Claude inserted literal control
characters (newlines, tabs) inside JSON string values instead of \\n / \\t.
"""

import json
import re
import sys
from pathlib import Path


def repair_line(raw: str) -> dict:
    """
    Attempt multiple repair strategies on a malformed JSON line.
    Returns parsed dict if successful, raises on complete failure.
    """
    # Strategy 1: try as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: replace literal control characters OUTSIDE of already-escaped sequences
    # Replace actual \n, \r, \t that appear raw inside strings
    cleaned = re.sub(r'(?<!\\)\n', '\\n', raw)
    cleaned = re.sub(r'(?<!\\)\r', '\\r', cleaned)
    cleaned = re.sub(r'(?<!\\)\t', '\\t', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: collapse the whole thing to one line, stripping stray newlines
    one_line = " ".join(raw.splitlines())
    try:
        return json.loads(one_line)
    except json.JSONDecodeError:
        pass

    # Strategy 4: strip trailing commas in arrays/objects (common Claude mistake)
    no_trailing = re.sub(r',\s*([\]}])', r'\1', one_line)
    try:
        return json.loads(no_trailing)
    except json.JSONDecodeError as e:
        raise ValueError(f"All repair strategies failed: {e}") from e


def fix_file(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)

    fixed = []
    errors = []

    with open(input_file, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Split on lines that START with {"proposal_id" to handle multi-line records
    # (in case Claude broke records across multiple lines)
    lines = raw_content.split('\n')

    # Re-join lines that are continuation of a broken JSON object
    merged_lines = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue

        if stripped.startswith('{"proposal_id"') and buffer:
            merged_lines.append(buffer)
            buffer = stripped
        elif stripped.startswith('{"proposal_id"'):
            buffer = stripped
        else:
            buffer = buffer + " " + stripped if buffer else stripped

    if buffer:
        merged_lines.append(buffer)

    print(f"Found {len(merged_lines)} records to process...\n")

    for i, line in enumerate(merged_lines, 1):
        try:
            obj = repair_line(line)

            # Also validate the nested assistant JSON string
            assistant_content = obj["messages"][2]["content"]
            nested = json.loads(assistant_content)

            # Re-serialise cleanly
            fixed.append(json.dumps(obj, ensure_ascii=False))
            print(f"Line {i}: ✅  {obj.get('proposal_id', 'UNKNOWN')}")
        except Exception as e:
            errors.append((i, str(e)))
            print(f"Line {i}: ❌  {e}")

    # Write fixed file
    with open(output_file, "w", encoding="utf-8") as f:
        for record in fixed:
            f.write(record + "\n")

    print(f"\n{'='*50}")
    print(f"✅  Fixed: {len(fixed)} records  →  {output_file}")
    print(f"❌  Failed: {len(errors)} records (need manual review)")
    if errors:
        print("\nFailed lines:")
        for line_no, err in errors:
            print(f"  Line {line_no}: {err}")


if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/processed/startup_proposals_claude_batch1.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".jsonl", "_fixed.jsonl")
    fix_file(input_path, output_path)
