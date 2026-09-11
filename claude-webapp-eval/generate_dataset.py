"""
generate_dataset.py
-------------------
Uses Claude to auto-generate an evaluation dataset for prompt testing.
Adapted for IBM endpoint (no assistant prefill, no temperature).

Usage:
    python generate_dataset.py
Outputs:
    dataset.json
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL   = os.getenv("ANTHROPIC_BASE_URL", "https://api.servicesessentials.ibm.com")
AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
MODEL      = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

MESSAGES_URL = f"{BASE_URL.rstrip('/')}/v1/messages"

API_HEADERS = {
    "x-api-key":          AUTH_TOKEN,
    "anthropic-version":  "2023-06-01",
    "content-type":       "application/json",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def chat(messages, system=None, stop_sequences=None, max_tokens=2048):
    """Non-streaming call to IBM/Claude endpoint."""
    payload = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "messages":   messages,
    }
    if system:
        payload["system"] = system
    if stop_sequences:
        payload["stop_sequences"] = stop_sequences

    resp = requests.post(MESSAGES_URL, headers=API_HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def extract_json(text):
    """Extract JSON array from a response that may be wrapped in markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last ``` line
        lines = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(lines).strip()
    return json.loads(text)


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n=5, topic="AWS-related tasks"):
    """
    Ask Claude to generate `n` eval test cases for the given topic.
    Returns a list of dicts: [{"task": "...", "expected_output_type": "..."}, ...]
    """
    prompt = f"""
Generate an evaluation dataset for a prompt evaluation tool. 
The dataset will test prompts that generate Python, JSON, or Regex for {topic}.

Return ONLY a valid JSON array — no explanation, no markdown fences.

Each object must have:
  "task"                - clear description of what to generate
  "expected_keyword"    - a word/pattern that MUST appear in a correct response
  "output_type"         - one of: python | json | regex

Rules:
- Focus on tasks solvable with a single Python function, single JSON object, or single regex
- Keep tasks concise and unambiguous
- Make expected_keyword something that will definitely appear in any correct answer

Generate exactly {n} objects.
"""
    messages = []
    add_user_message(messages, prompt)

    # IBM workaround: inject prefill as instruction instead of assistant role
    # Ask Claude to start its response directly with [
    messages[0]["content"] += "\n\nStart your response directly with [ (the opening bracket of the JSON array)."

    text = chat(messages, stop_sequences=None, max_tokens=2048)
    print(f"[RAW RESPONSE]\n{text[:500]}\n")
    return extract_json(text)


def generate_graded_dataset(n=5, topic="AWS-related tasks"):
    """
    Extended version: also generates expected output and grading method per case.
    Returns list of dicts ready to POST to /run_case.
    """
    prompt = f"""
Generate an evaluation dataset for testing an AI assistant on {topic}.

Return ONLY a valid JSON array — no explanation, no markdown fences.

Each object must have:
  "prompt"        - the exact question/instruction to send to the AI
  "expected"      - a keyword or short phrase that must appear in a correct response
  "method"        - grading method: "contains" | "exact" | "llm"
  "note"          - one sentence explaining why this is a good test case

Rules:
- Use "contains" for factual or code tasks where a keyword confirms correctness
- Use "exact" only for single-word or numeric answers
- Use "llm" for open-ended or quality-based tasks
- Mix output types: some Python, some JSON, some regex, some explanation tasks

Generate exactly {n} objects.
"""
    messages = []
    add_user_message(messages, prompt)
    messages[0]["content"] += "\n\nStart your response directly with [ (the opening bracket of the JSON array)."

    text = chat(messages, max_tokens=2048)
    print(f"[RAW RESPONSE]\n{text[:800]}\n")
    return extract_json(text)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Generating basic dataset (task + expected_keyword)...")
    print("=" * 60)
    basic = generate_dataset(n=5, topic="AWS-related tasks")
    print(f"\nOK Generated {len(basic)} basic test cases:")
    for i, item in enumerate(basic, 1):
        print(f"  {i}. [{item.get('output_type','?')}] {item.get('task','?')}")
        print(f"       expected_keyword: {item.get('expected_keyword','?')}")

    print("\n" + "=" * 60)
    print("Generating graded dataset (ready for /run_case)...")
    print("=" * 60)
    graded = generate_graded_dataset(n=5, topic="AWS-related tasks")
    print(f"\nOK Generated {len(graded)} graded test cases:")
    for i, item in enumerate(graded, 1):
        print(f"  {i}. [{item.get('method','?')}] {item.get('prompt','?')[:70]}")
        print(f"       expected: {item.get('expected','?')}")

    # Save both to file
    output = {
        "basic":  basic,
        "graded": graded,
    }
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to dataset.json")
    print("\nThe 'graded' array can be copy-pasted directly into the eval UI as test cases.")
