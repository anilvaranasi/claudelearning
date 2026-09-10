import os
import json
import uuid
from dotenv import load_dotenv
import anthropic
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, make_response

# Load variables from .env file
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

# Anthropic SDK client pointed at the IBM endpoint
client = anthropic.Anthropic(
    base_url=BASE_URL,
    api_key=AUTH_TOKEN,
)

def extract_json(text):
    """Extract and parse JSON from a response, stripping code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


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

    def generate():
        full_response = []
        try:
            print(f"\n{'='*60}")
            print(f"[SDK] [REQUEST] session:       {sid[:8]}...")
            print(f"[SDK] [REQUEST] turns:         {len(history)}")
            print(f"[SDK] [REQUEST] prompt:        {prompt!r}")
            print(f"[SDK] [REQUEST] prefill:       {prefill!r}")
            print(f"[SDK] [REQUEST] stop_seqs:     {stop_sequences}")
            print(f"[SDK] [REQUEST] system:        {system_prompt!r}")
            print(f"[SDK] [REQUEST] model:         {model}")
            print(f"{'='*60}")

            stream_kwargs = dict(
                model=model,
                max_tokens=4096,
                messages=history,
            )
            if system_prompt:
                stream_kwargs["system"] = system_prompt
            if stop_sequences:
                stream_kwargs["stop_sequences"] = stop_sequences

            with client.messages.stream(**stream_kwargs) as stream:
                model_emitted = False
                for text in stream.text_stream:
                    if not model_emitted:
                        model_name = getattr(stream.current_message_snapshot, "model", model)
                        print(f"[SDK] [RESPONSE] model: {model_name}")
                        yield f"data: {json.dumps({'type': 'model', 'model': model_name})}\n\n"
                        model_emitted = True
                    full_response.append(text)
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

                # Token usage — inside the with block, after text_stream is exhausted
                final_msg     = stream.get_final_message()
                input_tokens  = final_msg.usage.input_tokens
                output_tokens = final_msg.usage.output_tokens
                print(f"[SDK] [TOKENS] input={input_tokens} output={output_tokens} total={input_tokens + output_tokens}")
                yield f"data: {json.dumps({'type': 'usage', 'input': input_tokens, 'output': output_tokens})}\n\n"

            # Persist completed assistant reply
            assistant_text = "".join(full_response)
            history.append({"role": "assistant", "content": assistant_text})
            conversations[sid] = history
            print(f"[SDK] [HISTORY] saved {len(history)} messages for session {sid[:8]}...")
            print(f"[SDK] [RESPONSE] full text:\n{assistant_text}")

            # Attempt JSON extraction if prefill suggests structured output
            if prefill:
                try:
                    clean_json = extract_json(assistant_text)
                    pretty = json.dumps(clean_json, indent=2)
                    print(f"[SDK] [JSON] parsed successfully")
                    yield f"data: {json.dumps({'type': 'json_result', 'json': pretty})}\n\n"
                except (json.JSONDecodeError, ValueError):
                    print(f"[SDK] [JSON] response is not valid JSON — skipping")

            print(f"{'='*60}\n")
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except anthropic.APIStatusError as e:
            msg = f"API error {e.status_code}: {e.message}"
            print(f"[SDK] [ERROR] {msg}")
            yield f"data: {json.dumps({'type': 'error', 'text': msg})}\n\n"
        except Exception as e:
            print(f"[SDK] [ERROR] {e}")
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
    app.run(debug=True, port=5001)
