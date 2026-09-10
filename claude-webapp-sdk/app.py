import os
import json
import uuid
from dotenv import load_dotenv
import anthropic
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, make_response

# Load variables from .env file
load_dotenv()

app = Flask(__name__)

BASE_URL   = os.getenv("ANTHROPIC_BASE_URL", "https://api.servicesessentials.ibm.com")
AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
MODEL      = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# Anthropic SDK client pointed at the IBM endpoint
client = anthropic.Anthropic(
    base_url=BASE_URL,
    api_key=AUTH_TOKEN,
)

# In-memory conversation store: { session_id: [ {role, content}, ... ] }
conversations = {}


def get_session_id():
    """Return existing session cookie value or None."""
    return request.cookies.get("chat_session")


@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    # Assign a session ID cookie if the browser doesn't have one yet
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

    sid = get_session_id()
    if not sid:
        return jsonify({"error": "No session cookie. Please refresh the page."}), 400

    # Load history, append new user message
    history = conversations.get(sid, [])
    history.append({"role": "user", "content": prompt})

    def generate():
        full_response = []
        try:
            print(f"\n{'='*60}")
            print(f"[SDK] [REQUEST] session: {sid[:8]}...")
            print(f"[SDK] [REQUEST] turns:   {len(history)}")
            print(f"[SDK] [REQUEST] prompt:  {prompt!r}")
            print(f"[SDK] [REQUEST] model:   {MODEL}")
            print(f"{'='*60}")

            with client.messages.stream(
                model=MODEL,
                max_tokens=4096,
                messages=history,
            ) as stream:
                model_emitted = False
                for text in stream.text_stream:
                    if not model_emitted:
                        model_name = getattr(stream.current_message_snapshot, "model", MODEL)
                        print(f"[SDK] [RESPONSE] model: {model_name}")
                        yield f"data: {json.dumps({'type': 'model', 'model': model_name})}\n\n"
                        model_emitted = True
                    full_response.append(text)
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

            # Persist completed assistant reply into in-memory store
            assistant_text = "".join(full_response)
            history.append({"role": "assistant", "content": assistant_text})
            conversations[sid] = history
            print(f"[SDK] [HISTORY] saved {len(history)} messages for session {sid[:8]}...")

            # Token usage — safe to call after text_stream is fully consumed
            final_msg     = stream.get_final_message()
            input_tokens  = final_msg.usage.input_tokens
            output_tokens = final_msg.usage.output_tokens
            print(f"[SDK] [TOKENS] input={input_tokens} output={output_tokens} total={input_tokens + output_tokens}")
            yield f"data: {json.dumps({'type': 'usage', 'input': input_tokens, 'output': output_tokens})}\n\n"

            print(f"[SDK] [RESPONSE] full text:\n{assistant_text}")
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
