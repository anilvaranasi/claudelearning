import os
import json
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import anthropic
from flask import Flask, render_template, request, jsonify, make_response

load_dotenv()

app = Flask(__name__)

BASE_URL      = os.getenv("ANTHROPIC_BASE_URL", "https://api.servicesessentials.ibm.com")
AUTH_TOKEN    = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

AVAILABLE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
]

client = anthropic.Anthropic(
    base_url=BASE_URL,
    api_key=AUTH_TOKEN,
)

# ── In-memory conversation store ──────────────────────────────────────────────
conversations = {}   # { session_id: [message, ...] }


# ── Tool implementations (from 001_tools.ipynb) ───────────────────────────────

def add_duration_to_datetime(
    datetime_str, duration=0, unit="days", input_format="%Y-%m-%d"
):
    """Add a duration to a datetime string and return a human-readable result."""
    date = datetime.strptime(datetime_str, input_format)

    if unit == "seconds":
        new_date = date + timedelta(seconds=duration)
    elif unit == "minutes":
        new_date = date + timedelta(minutes=duration)
    elif unit == "hours":
        new_date = date + timedelta(hours=duration)
    elif unit == "days":
        new_date = date + timedelta(days=duration)
    elif unit == "weeks":
        new_date = date + timedelta(weeks=duration)
    elif unit == "months":
        month = date.month + duration
        year  = date.year + month // 12
        month = month % 12
        if month == 0:
            month = 12
            year -= 1
        days_in_month = [
            31,
            29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
        ]
        day = min(date.day, days_in_month[month - 1])
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        new_date = date.replace(year=date.year + duration)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    """Return the current date/time formatted according to date_format."""
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)


def set_reminder(content, timestamp):
    """Simulate setting a reminder — returns confirmation string."""
    return f"Reminder set for {timestamp}: {content}"


def handle_batch_tool(invocations):
    """
    Execute multiple tool calls at once (batch_tool meta-tool).
    Each invocation: {"name": "...", "arguments": "<json string>"}
    Returns list of individual results.
    """
    results = []
    for inv in invocations:
        name = inv.get("name")
        try:
            args = json.loads(inv.get("arguments", "{}"))
        except json.JSONDecodeError as e:
            results.append({"tool": name, "error": f"Invalid JSON arguments: {e}"})
            continue
        result = dispatch_tool(name, args)
        results.append({"tool": name, "result": result})
    return results


def dispatch_tool(name, args):
    """Route a tool call by name to its implementation."""
    if name == "get_current_datetime":
        return get_current_datetime(
            date_format=args.get("date_format", "%Y-%m-%d %H:%M:%S"),
        )
    elif name == "add_duration_to_datetime":
        return add_duration_to_datetime(
            datetime_str  = args["datetime_str"],
            duration      = args.get("duration", 0),
            unit          = args.get("unit", "days"),
            input_format  = args.get("input_format", "%Y-%m-%d"),
        )
    elif name == "set_reminder":
        return set_reminder(
            content   = args["content"],
            timestamp = args["timestamp"],
        )
    elif name == "batch_tool":
        return handle_batch_tool(args.get("invocations", []))
    else:
        raise ValueError(f"Unknown tool: {name}")


# ── Tool schemas (from 001_tools.ipynb) ───────────────────────────────────────

ADD_DURATION_SCHEMA = {
    "name": "add_duration_to_datetime",
    "description": (
        "Adds a specified duration to a datetime string and returns the resulting datetime "
        "in a detailed format. Handles seconds, minutes, hours, days, weeks, months, and years. "
        "Output format: 'Thursday, April 03, 2025 10:30:00 AM'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {
                "type": "string",
                "description": "The input datetime string, formatted according to input_format.",
            },
            "duration": {
                "type": "number",
                "description": "Amount of time to add (positive = future, negative = past). Defaults to 0.",
            },
            "unit": {
                "type": "string",
                "description": "Time unit: 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', or 'years'. Defaults to 'days'.",
            },
            "input_format": {
                "type": "string",
                "description": "Python strptime format for datetime_str, e.g. '%Y-%m-%d'. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
}

SET_REMINDER_SCHEMA = {
    "name": "set_reminder",
    "description": (
        "Creates a timed reminder that will notify the user at the specified time. "
        "Use when the user wants to be reminded about something at a future point in time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The reminder message, e.g. 'Take medication' or 'Join team call'.",
            },
            "timestamp": {
                "type": "string",
                "description": "When to trigger the reminder (ISO 8601: YYYY-MM-DDTHH:MM:SS or Unix timestamp).",
            },
        },
        "required": ["content", "timestamp"],
    },
}

BATCH_TOOL_SCHEMA = {
    "name": "batch_tool",
    "description": "Invoke multiple other tool calls simultaneously.",
    "input_schema": {
        "type": "object",
        "properties": {
            "invocations": {
                "type": "array",
                "description": "The tool calls to invoke.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":      {"type": "string", "description": "Tool name to invoke."},
                        "arguments": {"type": "string", "description": "Tool arguments as a JSON string."},
                    },
                    "required": ["name", "arguments"],
                },
            }
        },
        "required": ["invocations"],
    },
}

GET_CURRENT_DATETIME_SCHEMA = {
    "name": "get_current_datetime",
    "description": (
        "Returns the current date and time formatted according to the specified format string. "
        "Use this when you need to know the current date and time, such as for timestamping records, "
        "calculating time differences, or displaying the current time to users. "
        "Default format: '%Y-%m-%d %H:%M:%S' returns a timestamp like '2025-05-07 14:32:15'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": (
                    "Python strftime format string. E.g. '%Y-%m-%d' for date only, "
                    "'%H:%M:%S' for time only, '%B %d, %Y' for 'May 07, 2025'. "
                    "Defaults to '%Y-%m-%d %H:%M:%S'."
                ),
            },
        },
        "required": [],
    },
}

ALL_TOOLS = [GET_CURRENT_DATETIME_SCHEMA, ADD_DURATION_SCHEMA, SET_REMINDER_SCHEMA, BATCH_TOOL_SCHEMA]


# ── Agentic tool-use loop ─────────────────────────────────────────────────────

def run_agent_loop(messages, model, system=None):
    """
    Agentic loop: call Claude, execute any tool calls, feed results back,
    repeat until stop_reason == 'end_turn'.

    Returns a list of turn events for the UI:
      {"type": "text",        "text": "..."}
      {"type": "tool_call",   "name": "...", "input": {...}, "id": "..."}
      {"type": "tool_result", "name": "...", "result": "...", "id": "..."}
    """
    events = []

    while True:
        kwargs = dict(
            model      = model,
            max_tokens = 4096,
            tools      = ALL_TOOLS,
            messages   = messages,
        )
        if system:
            kwargs["system"] = system

        print(f"\n[TOOLS] Calling Claude ({model}), {len(messages)} messages in history")
        response = client.messages.create(**kwargs)
        print(f"[TOOLS] stop_reason={response.stop_reason}  blocks={len(response.content)}")

        # Collect text + tool_use blocks from this response
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                print(f"[TOOLS] text: {block.text[:120]!r}")
                events.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                print(f"[TOOLS] tool_use: {block.name}({json.dumps(block.input)[:120]})")
                events.append({
                    "type":  "tool_call",
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })
                tool_uses.append(block)

        # Add Claude's full response turn to history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls or done, stop
        if response.stop_reason == "end_turn" or not tool_uses:
            break

        # Execute all tool calls and build tool_result content block list
        tool_results = []
        for tu in tool_uses:
            try:
                result = dispatch_tool(tu.name, tu.input)
                result_str = json.dumps(result) if not isinstance(result, str) else result
                print(f"[TOOLS] result for {tu.name}: {result_str[:120]!r}")
                events.append({
                    "type":   "tool_result",
                    "id":     tu.id,
                    "name":   tu.name,
                    "result": result_str,
                    "error":  False,
                })
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tu.id,
                    "content":     result_str,
                })
            except Exception as e:
                err_str = str(e)
                print(f"[TOOLS] error in {tu.name}: {err_str}")
                events.append({
                    "type":   "tool_result",
                    "id":     tu.id,
                    "name":   tu.name,
                    "result": err_str,
                    "error":  True,
                })
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tu.id,
                    "content":     err_str,
                    "is_error":    True,
                })

        # Feed tool results back for next loop iteration
        messages.append({"role": "user", "content": tool_results})

    return events


# ── Flask routes ──────────────────────────────────────────────────────────────

def get_session_id():
    return request.cookies.get("tools_session")


@app.route("/")
def index():
    resp = make_response(render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL,
    ))
    if not request.cookies.get("tools_session"):
        resp.set_cookie("tools_session", str(uuid.uuid4()), samesite="Lax")
    return resp


@app.route("/clear", methods=["POST"])
def clear():
    sid = get_session_id()
    if sid and sid in conversations:
        del conversations[sid]
    return jsonify({"status": "cleared"})


@app.route("/chat", methods=["POST"])
def chat():
    data   = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    model  = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    system = data.get("system_prompt", "").strip() or None

    sid = get_session_id()
    if not sid:
        return jsonify({"error": "No session cookie. Please refresh."}), 400

    history = conversations.get(sid, [])
    history.append({"role": "user", "content": prompt})

    try:
        events = run_agent_loop(history, model, system)
        # history was mutated in-place by run_agent_loop; save it back
        conversations[sid] = history
        return jsonify({"events": events})
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"API error {e.status_code}: {e.message}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5004)
