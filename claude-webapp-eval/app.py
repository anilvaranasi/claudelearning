import os
import json
import uuid
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, make_response

load_dotenv()

app = Flask(__name__)

BASE_URL      = os.getenv("ANTHROPIC_BASE_URL", "https://api.servicesessentials.ibm.com")
AUTH_TOKEN    = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

AVAILABLE_MODELS = [
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5",
]

MESSAGES_URL = f"{BASE_URL.rstrip('/')}/v1/messages"

API_HEADERS = {
    "x-api-key": AUTH_TOKEN,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

JUDGE_SYSTEM = """You are an impartial evaluator grading an AI assistant's response.
You will be given a question, an expected answer, and the actual response.
Grade the response and return ONLY valid JSON — no explanation outside the JSON.
Schema: {"score": <1-5>, "pass": <true|false>, "reasoning": "<one sentence>"}
pass=true if score >= 3."""


def call_claude(messages, system=None, model=None, max_tokens=1024):
    """Non-streaming Claude call. Returns response text or raises."""
    payload = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    resp = requests.post(MESSAGES_URL, headers=API_HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


@app.route("/")
def index():
    resp = make_response(render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL,
    ))
    if not request.cookies.get("eval_session"):
        resp.set_cookie("eval_session", str(uuid.uuid4()), samesite="Lax")
    return resp


@app.route("/run_case", methods=["POST"])
def run_case():
    """Run a single test case and return graded result."""
    data        = request.get_json()
    prompt      = data.get("prompt", "").strip()
    expected    = data.get("expected", "").strip()
    method      = data.get("method", "contains")   # exact | contains | llm
    system      = data.get("system_prompt", "").strip()
    model       = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    print(f"\n[EVAL] prompt={prompt!r} method={method} model={model}")

    try:
        # Step 1 — get Claude's response to the prompt
        messages = [{"role": "user", "content": prompt}]
        response = call_claude(messages, system=system or None, model=model)
        print(f"[EVAL] response={response[:120]!r}...")

        # Step 2 — grade the response
        score = None
        passed = None
        reasoning = ""

        if method == "exact":
            passed = response.strip().lower() == expected.strip().lower()
            score = 5 if passed else 1
            reasoning = "Exact match." if passed else "Response did not exactly match expected."

        elif method == "contains":
            passed = expected.lower() in response.lower()
            score = 5 if passed else 1
            reasoning = f"Expected text {'found' if passed else 'not found'} in response."

        elif method == "llm":
            judge_prompt = f"""Question: {prompt}

Expected answer: {expected}

Actual response: {response}

Grade the actual response."""
            judge_messages = [{"role": "user", "content": judge_prompt}]
            judge_resp = call_claude(judge_messages, system=JUDGE_SYSTEM, model=model, max_tokens=256)
            print(f"[EVAL] judge_resp={judge_resp!r}")
            # Extract JSON from judge response
            judge_text = judge_resp.strip()
            if judge_text.startswith("```"):
                lines = judge_text.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                judge_text = "\n".join(lines).strip()
            verdict = json.loads(judge_text)
            score     = verdict.get("score", 3)
            passed    = verdict.get("pass", score >= 3)
            reasoning = verdict.get("reasoning", "")

        print(f"[EVAL] score={score} pass={passed}")

        return jsonify({
            "prompt":    prompt,
            "expected":  expected,
            "response":  response,
            "score":     score,
            "passed":    passed,
            "reasoning": reasoning,
            "method":    method,
            "model":     model,
        })

    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5003)
