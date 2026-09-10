import os
import json
import uuid
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, make_response

# Load variables from .env file
load_dotenv()

app = Flask(__name__)

BASE_URL   = os.getenv("ANTHROPIC_BASE_URL", "https://api.servicesessentials.ibm.com")
AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

AVAILABLE_MODELS = [
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5",
]

MESSAGES_URL = f"{BASE_URL.rstrip('/')}/v1/messages"


def extract_json(text):
    """Extract and parse JSON from a response, stripping code fences if present."""
    # Strip ```json ... ``` or ``` ... ``` fences
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Remove opening fence line (```json or ```)
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)

API_HEADERS = {
    "x-api-key": AUTH_TOKEN,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# In-memory conversation store: { session_id: [ {role, content}, ... ] }
conversations = {}


def get_session_id():
    """Return existing session cookie value or None."""
    return request.cookies.get("chat_session")


@app.route("/")
def index():
    resp = make_response(render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL,
    ))
    if not request.cookies.get("chat_session"):
        resp.set_cookie("chat_session", str(uuid.uuid4()), samesite="Lax")
    return resp


@app.route("/clear", methods=["POST"])
def clear():
    sid = get_session_id()
    if sid and sid in conversations:
        del conversations[sid]
    return jsonify({"status": "cleared"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    model          = data.get("model", DEFAULT_MODEL).strip()
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    system_prompt  = data.get("system_prompt", "").strip()
    prefill        = data.get("prefill", "").strip()
    stop_sequences = [s.strip() for s in data.get("stop_sequences", "").split(",") if s.strip()]

    sid = get_session_id()
    if not sid:
        return jsonify({"error": "No session cookie. Please refresh the page."}), 400

    # Load history, append new user message
    history = conversations.get(sid, [])

    # Prefill: append as suffix to user message (IBM endpoint doesn't support assistant prefill)
    user_content = prompt
    if prefill:
        user_content = f"{prompt}\n\nRespond starting with: {prefill}"
    history.append({"role": "user", "content": user_content})

    payload = {
        "model": model,
        "max_tokens": 4096,
        "stream": True,
        "messages": history,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if stop_sequences:
        payload["stop_sequences"] = stop_sequences

    def generate():
        full_response = []
        try:
            print(f"\n{'='*60}")
            print(f"[DOTENV] [REQUEST] session: {sid[:8]}...")
            print(f"[DOTENV] [REQUEST] turns:   {len(history)}")
            print(f"[DOTENV] [REQUEST] prompt:         {prompt!r}")
            print(f"[DOTENV] [REQUEST] prefill:        {prefill!r}")
            print(f"[DOTENV] [REQUEST] stop_sequences: {stop_sequences}")
            print(f"[DOTENV] [REQUEST] system:         {system_prompt!r}")
            print(f"[DOTENV] [REQUEST] model:          {model}")
            print(f"{'='*60}")

            with requests.post(
                MESSAGES_URL,
                headers=API_HEADERS,
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                print(f"[DOTENV] [RESPONSE] status: {resp.status_code}")

                if not resp.ok:
                    error_body = resp.text
                    print(f"[DOTENV] [RESPONSE] error: {error_body}")
                    yield f"data: {json.dumps({'type': 'error', 'text': f'API error {resp.status_code}: {error_body}'})}\n\n"
                    return

                model_sent = False
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if not line.startswith("data:"):
                        continue

                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if not model_sent and event_type == "message_start":
                        model_name = event.get("message", {}).get("model", model)
                        print(f"[DOTENV] [RESPONSE] model: {model_name}")
                        yield f"data: {json.dumps({'type': 'model', 'model': model_name})}\n\n"
                        model_sent = True

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            full_response.append(text)
                            yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

                    if event_type == "message_delta":
                        usage = event.get("usage", {})
                        input_tokens  = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        print(f"[DOTENV] [TOKENS] input={input_tokens} output={output_tokens} total={input_tokens + output_tokens}")
                        yield f"data: {json.dumps({'type': 'usage', 'input': input_tokens, 'output': output_tokens})}\n\n"

            # Persist completed assistant reply
            assistant_text = "".join(full_response)
            history.append({"role": "assistant", "content": assistant_text})
            conversations[sid] = history
            print(f"[DOTENV] [HISTORY] saved {len(history)} messages for session {sid[:8]}...")
            print(f"[DOTENV] [RESPONSE] full text:\n{assistant_text}")

            # Attempt JSON extraction if prefill suggests structured output
            if prefill:
                try:
                    clean_json = extract_json(assistant_text)
                    pretty = json.dumps(clean_json, indent=2)
                    print(f"[DOTENV] [JSON] parsed successfully")
                    yield f"data: {json.dumps({'type': 'json_result', 'json': pretty})}\n\n"
                except (json.JSONDecodeError, ValueError):
                    print(f"[DOTENV] [JSON] response is not valid JSON — skipping")

            print(f"{'='*60}\n")

        except Exception as e:
            print(f"[DOTENV] [ERROR] {e}")
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)
