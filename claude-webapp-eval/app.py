import os
import re
import json
import uuid
import requests
from textwrap import dedent
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

AWS_JUDGE_SYSTEM = """You are an expert AWS code reviewer. Evaluate the AI-generated solution.
Return ONLY a valid JSON object — no prose, no markdown fences, nothing else.
Use this exact schema:
{"strengths": ["..."], "weaknesses": ["..."], "reasoning": "...", "score": 7}
Keep strengths and weaknesses to 1-3 items each. score must be an integer 1-10."""


def _extract_json(text):
    """
    Robustly extract a JSON object or array from a model response.
    Handles: plain JSON, ```json fences, prose before/after JSON.
    """
    text = text.strip()
    # Strip markdown fences first
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        lines = [l for l in lines if l.strip() not in ("```", "```json")]
        text = "\n".join(lines).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the outermost { ... } or [ ... ]
    for start_char, end_char in (('{', '}'), ('[', ']')):
        start = text.find(start_char)
        end   = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"No valid JSON found in model response: {text[:200]!r}")


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
    print(f"[API] status={resp.status_code} body={resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()
    # Guard: some IBM gateway errors return 200 with no content array
    if "content" not in data or not data["content"]:
        raise ValueError(f"Unexpected API response (no content): {resp.text[:300]}")
    # Find the first text block — model may return thinking blocks before the text
    for block in data["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise ValueError(f"No text block found in response content: {data['content']}")


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
            verdict = _extract_json(judge_resp)
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


@app.route("/generate_dataset", methods=["POST"])
def generate_dataset():
    """Ask Claude to generate AWS eval test cases (notebook-style dataset generation)."""
    data  = request.get_json()
    n     = int(data.get("n", 3))
    topic = data.get("topic", "AWS-related tasks").strip() or "AWS-related tasks"
    model = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL

    prompt = f"""Generate an evaluation dataset for a prompt evaluation tool.
The dataset will test prompts that generate Python, JSON, or Regex for {topic}.

Return ONLY a valid JSON array — no explanation, no markdown fences.

Each object must have:
  "task"             - clear description of what to generate
  "format"           - one of: python | json | regex

Rules:
- Focus on tasks solvable with a single Python function, single JSON object, or single regex
- Keep tasks concise and unambiguous
- Do not require writing much code

Generate exactly {n} objects.
Start your response directly with [ (the opening bracket of the JSON array)."""

    try:
        messages = [{"role": "user", "content": prompt}]
        text = call_claude(messages, model=model, max_tokens=2048)
        dataset = _extract_json(text)
        return jsonify({"dataset": dataset})
    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run_aws_eval", methods=["POST"])
def run_aws_eval():
    """
    Notebook-style eval: send a task to Claude, then grade with the AWS judge
    returning strengths/weaknesses/reasoning/score (1-10).
    """
    data  = request.get_json()
    task  = data.get("task", "").strip()
    model = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL

    if not task:
        return jsonify({"error": "task is required"}), 400

    try:
        # Step 1 — run the task through Claude
        messages = [{"role": "user", "content": f"Please solve the following task:\n\n{task}"}]
        output = call_claude(messages, model=model)

        # Step 2 — grade with AWS judge
        # Keep task + solution in XML tags so any curly braces inside code don't
        # interfere with JSON parsing of the judge response.
        eval_prompt = (
            "Original Task:\n<task>\n" + task + "\n</task>\n\n"
            "Solution to Evaluate:\n<solution>\n" + output + "\n</solution>\n\n"
            "Grade the solution. Return ONLY a JSON object."
        )

        judge_messages = [{"role": "user", "content": eval_prompt}]
        judge_text = call_claude(judge_messages, system=AWS_JUDGE_SYSTEM, model=model, max_tokens=512)
        grade = _extract_json(judge_text)

        return jsonify({
            "task":       task,
            "output":     output,
            "score":      grade.get("score", 5),
            "strengths":  grade.get("strengths", []),
            "weaknesses": grade.get("weaknesses", []),
            "reasoning":  grade.get("reasoning", ""),
            "model":      model,
        })

    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── PromptEvaluator helpers (ported from 001_prompting.ipynb) ─────────────────

def _render_template(template_string, variables):
    """Replace {placeholder} tokens; {{ and }} become literal braces."""
    placeholders = re.findall(r"\{([^{}]+)\}", template_string)
    result = template_string
    for ph in placeholders:
        if ph in variables:
            result = result.replace("{" + ph + "}", str(variables[ph]))
    return result.replace("{{", "{").replace("}}", "}")


def _strip_json_fence(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        lines = [l for l in lines if l.strip() != "```"]
        text = "\n".join(lines).strip()
    return text


@app.route("/pe/generate_ideas", methods=["POST"])
def pe_generate_ideas():
    """Step 1 – generate N diverse scenario ideas for the task."""
    data              = request.get_json()
    task_description  = data.get("task_description", "").strip()
    prompt_inputs_spec= data.get("prompt_inputs_spec", {})  # {key: description}
    num_cases         = int(data.get("num_cases", 3))
    model             = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    if not task_description:
        return jsonify({"error": "task_description is required"}), 400

    example_inputs = ""
    for key, value in prompt_inputs_spec.items():
        val = value.replace("\n", "\\n")
        example_inputs += f'"{key}": str # {val},'

    prompt = dedent(f"""
        Generate {num_cases} unique, diverse ideas for testing a prompt that accomplishes this task:

        <task_description>
        {task_description}
        </task_description>

        The prompt will receive the following inputs
        <prompt_inputs>
        {example_inputs}
        </prompt_inputs>

        Each idea should represent a distinct scenario or example that tests different aspects of the task.

        Output Format:
        Provide your response as a JSON array where each item is a brief description of the idea.

        Ensure each idea is:
        - Clearly distinct from the others
        - Relevant to the task description
        - Specific enough to guide generation of a full test case
        - Quick to solve without requiring extensive computation or multi-step processing
        - Solvable with no more than 400 tokens of output

        Remember, only generate {num_cases} unique ideas.
        Start your response directly with [ (the opening bracket of the JSON array).
    """)

    system = "You are a test scenario designer specialized in creating diverse, unique testing scenarios."
    try:
        messages = [{"role": "user", "content": prompt}]
        text = call_claude(messages, system=system, model=model, max_tokens=2048)
        ideas = _extract_json(text)
        return jsonify({"ideas": ideas})
    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pe/generate_test_case", methods=["POST"])
def pe_generate_test_case():
    """Step 2 – generate a single test case (prompt_inputs + solution_criteria) for one idea."""
    data              = request.get_json()
    task_description  = data.get("task_description", "").strip()
    idea              = data.get("idea", "").strip()
    prompt_inputs_spec= data.get("prompt_inputs_spec", {})
    model             = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    if not task_description or not idea:
        return jsonify({"error": "task_description and idea are required"}), 400

    example_prompt_inputs = ""
    for key, value in prompt_inputs_spec.items():
        val = value.replace("\n", "\\n")
        example_prompt_inputs += f'"{key}": "EXAMPLE_VALUE", // {val}\n'
    allowed_keys = ", ".join([f'"{k}"' for k in prompt_inputs_spec.keys()])

    prompt = dedent(f"""
        Generate a single detailed test case for a prompt evaluation based on:

        <task_description>
        {task_description}
        </task_description>

        <specific_idea>
        {idea}
        </specific_idea>

        <allowed_input_keys>
        {allowed_keys}
        </allowed_input_keys>

        Output Format:
        Return ONLY valid JSON — no markdown fences.
        {{
            "prompt_inputs": {{
            {example_prompt_inputs}
            }},
            "solution_criteria": ["criterion 1", "criterion 2"]
        }}

        IMPORTANT REQUIREMENTS:
        - You MUST ONLY use these exact input keys in your prompt_inputs: {allowed_keys}
        - Do NOT add any additional keys to prompt_inputs
        - All keys listed in allowed_input_keys must be included
        - Include measurable, concise solution criteria (1 to 4 items)
        - The solution criteria should ONLY address the direct requirements of the task
        - Quick to solve, solvable with no more than 400 tokens of output
        - DO NOT include any fields beyond those specified

        Start your response directly with {{ (the opening brace of the JSON object).
    """)

    system = "You are a test case creator specializing in designing evaluation scenarios."
    try:
        messages = [{"role": "user", "content": prompt}]
        text = call_claude(messages, system=system, model=model, max_tokens=1024)
        test_case = _extract_json(text)
        test_case["task_description"] = task_description
        test_case["scenario"] = idea
        return jsonify({"test_case": test_case})
    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pe/run_test_case", methods=["POST"])
def pe_run_test_case():
    """Step 3 – run user's prompt template against a test case, then grade the output."""
    data           = request.get_json()
    test_case      = data.get("test_case", {})
    prompt_template= data.get("prompt_template", "").strip()
    extra_criteria = data.get("extra_criteria", "").strip()
    model          = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    if not test_case or not prompt_template:
        return jsonify({"error": "test_case and prompt_template are required"}), 400

    try:
        # Build the user prompt by substituting {variable} tokens from prompt_inputs
        prompt_inputs = test_case.get("prompt_inputs", {})
        user_prompt   = _render_template(prompt_template, prompt_inputs)

        # Run the prompt
        messages = [{"role": "user", "content": user_prompt}]
        output = call_claude(messages, model=model, max_tokens=1000)

        # Build grading inputs
        prompt_inputs_str = ""
        for key, value in prompt_inputs.items():
            val = str(value).replace("\n", "\\n")
            prompt_inputs_str += f'"{key}":"{val}",\n'

        extra_criteria_section = ""
        if extra_criteria:
            extra_criteria_section = (
                "\nMandatory Requirements - ANY VIOLATION MEANS AUTOMATIC FAILURE (score of 3 or lower):\n"
                "<extra_important_criteria>\n" + extra_criteria + "\n</extra_important_criteria>\n"
            )

        criteria_text = "\n".join(test_case.get("solution_criteria", []))

        # Build eval prompt using concatenation — avoids f-string curly-brace
        # collisions when output/inputs contain Python/JSON code with { } chars.
        eval_prompt = (
            "Your task is to evaluate the following AI-generated solution with EXTREME RIGOR.\n\n"
            "Original task description:\n<task_description>\n"
            + test_case.get("task_description", "") +
            "\n</task_description>\n\n"
            "Original task inputs:\n<task_inputs>\n" + prompt_inputs_str + "\n</task_inputs>\n\n"
            "Solution to Evaluate:\n<solution>\n" + output + "\n</solution>\n\n"
            "Criteria:\n<criteria>\n" + criteria_text + "\n</criteria>\n"
            + extra_criteria_section +
            "\nScoring Guidelines:\n"
            "* Score 1-3: fails mandatory requirements\n"
            "* Score 4-6: meets mandatory but significant deficiencies\n"
            "* Score 7-8: meets mandatory + most secondary, minor issues\n"
            "* Score 9-10: meets all criteria\n\n"
            "IMPORTANT: Grade ONLY on listed criteria. ANY mandatory violation = score 3 or lower.\n\n"
            'Return ONLY valid JSON: {"strengths":[...],"weaknesses":[...],"reasoning":"...","score":N}'
        )

        judge_messages = [{"role": "user", "content": eval_prompt}]
        judge_text = call_claude(judge_messages, model=model, max_tokens=512)
        grade = _extract_json(judge_text)

        return jsonify({
            "output":       output,
            "test_case":    test_case,
            "score":        grade.get("score", 5),
            "strengths":    grade.get("strengths", []),
            "weaknesses":   grade.get("weaknesses", []),
            "reasoning":    grade.get("reasoning", ""),
            "model":        model,
        })

    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pe/export_report", methods=["POST"])
def pe_export_report():
    """
    Port of generate_prompt_evaluation_report() from 001_prompting.ipynb.
    Accepts the full eval results array and returns a self-contained HTML report.
    """
    data    = request.get_json()
    results = data.get("results", [])
    if not results:
        return jsonify({"error": "results array is required"}), 400

    from statistics import mean as _mean
    total_tests = len(results)
    scores      = [r.get("score", 0) for r in results]
    avg_score   = _mean(scores) if scores else 0
    pass_rate   = 100 * len([s for s in scores if s >= 7]) / total_tests if total_tests else 0

    # Build table rows
    rows_html = ""
    for result in results:
        tc = result.get("test_case", {})
        prompt_inputs_html = "".join(
            f"<strong>{k}:</strong> {v}<br>"
            for k, v in (tc.get("prompt_inputs") or {}).items()
        )
        criteria_items = tc.get("solution_criteria") or []
        criteria_html  = ("• " + "<br>• ".join(criteria_items)) if criteria_items else "—"
        score = result.get("score", 0)
        score_class = "score-high" if score >= 8 else ("score-low" if score <= 5 else "score-medium")
        output_escaped = (result.get("output") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows_html += f"""
            <tr>
                <td>{tc.get("scenario", "—")}</td>
                <td class="prompt-inputs">{prompt_inputs_html}</td>
                <td class="criteria">{criteria_html}</td>
                <td class="output"><pre>{output_escaped}</pre></td>
                <td class="score-col"><span class="score {score_class}">{score}</span></td>
                <td class="reasoning">{result.get("reasoning", "—")}</td>
            </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary-stats {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; }}
        .stat-box {{ background: #fff; border-radius: 5px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,.1); flex-basis: 30%; min-width: 200px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #4a4a4a; color: #fff; text-align: left; padding: 12px; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; vertical-align: top; width: 20%; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .score {{ font-weight: bold; padding: 5px 10px; border-radius: 3px; display: inline-block; }}
        .score-high   {{ background: #c8e6c9; color: #2e7d32; }}
        .score-medium {{ background: #fff9c4; color: #f57f17; }}
        .score-low    {{ background: #ffcdd2; color: #c62828; }}
        .score-col {{ width: 80px; }}
        .output pre {{
            background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;
            padding: 10px; margin: 0; font-family: Consolas, Monaco, monospace;
            font-size: 13px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Prompt Evaluation Report</h1>
        <div class="summary-stats">
            <div class="stat-box">
                <div>Total Test Cases</div>
                <div class="stat-value">{total_tests}</div>
            </div>
            <div class="stat-box">
                <div>Average Score</div>
                <div class="stat-value">{avg_score:.1f} / 10</div>
            </div>
            <div class="stat-box">
                <div>Pass Rate (&ge;7)</div>
                <div class="stat-value">{pass_rate:.1f}%</div>
            </div>
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>Scenario</th>
                <th>Prompt Inputs</th>
                <th>Solution Criteria</th>
                <th>Output</th>
                <th>Score</th>
                <th>Reasoning</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""

    from flask import Response
    return Response(
        html,
        mimetype="text/html",
        headers={"Content-Disposition": "attachment; filename=prompt_eval_report.html"}
    )


if __name__ == "__main__":
    app.run(debug=True, port=5003)
