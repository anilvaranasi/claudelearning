import os, json, requests
from dotenv import load_dotenv
load_dotenv()

BASE_URL   = os.getenv("ANTHROPIC_BASE_URL")
AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN")
MODEL      = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
MESSAGES_URL = BASE_URL.rstrip("/") + "/v1/messages"
API_HEADERS = {
    "x-api-key": AUTH_TOKEN,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

prompt = """Generate an evaluation dataset for testing an AI assistant on Telugu Chandas (classical Telugu poetry meters).

Return ONLY a valid JSON array - no explanation, no markdown fences.

Each object must have:
  "prompt"    - the exact instruction to send to the AI
  "expected"  - a keyword or phrase that must appear in a correct response
  "method"    - grading method: "contains" or "llm"
  "category"  - one of: knowledge | generation | script | robustness

Cover these areas:
1. Meter knowledge - Kandapadyam, Utpalamala, Seesamala, Champakamala, Ataveladi rules
2. Telugu script output - verify response is in Telugu Unicode (not transliteration)
3. Prasa (rhyme) rules - aadi prasa, antyaprasa
4. Yati (caesura) placement rules
5. Gana structure - laghu/guru syllables
6. Matra counting
7. Generation quality - does generated poem follow the requested meter

Generate exactly 10 objects. Start your response directly with ["""

resp = requests.post(MESSAGES_URL, headers=API_HEADERS, json={
    "model": MODEL,
    "max_tokens": 3000,
    "messages": [{"role": "user", "content": prompt}]
}, timeout=120)
resp.raise_for_status()

text = resp.json()["content"][0]["text"].strip()
if text.startswith("```"):
    lines = text.split("\n")
    lines = [l for l in lines[1:] if l.strip() != "```"]
    text = "\n".join(lines).strip()

dataset = json.loads(text)

with open("dataset_chandas.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Generated {len(dataset)} test cases -> dataset_chandas.json\n")
for i, item in enumerate(dataset, 1):
    print(f"  {i}. [{item['category']}] [{item['method']}]")
    print(f"     prompt   : {item['prompt'][:80]}")
    print(f"     expected : {item['expected']}")
    print()
