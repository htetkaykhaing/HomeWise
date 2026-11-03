"""Supervisor router: turns a user question into a routing record.

Rules R1-R8: agents/supervisor/prompts/supervisor_v2.md
Code walkthrough: Docs/supervisor_guide.md
"""

import json
from copy import deepcopy
import os
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = Path(__file__).parent / "prompts" / "supervisor_v2.md"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class RouterReplyError(Exception):
    """The model replied, but with no usable content (e.g. a refusal)."""


def record_usage(usage_log, step, response):
    """Add one call's token counts to usage_log (a list), if one is given."""
    if usage_log is None:
        return
    usage = getattr(response, "usage", None)
    usage_log.append({
        "step": step,
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    })


def estimate_cost_usd(usage_log, pricing=None):
    """Estimate the USD cost of every call in a usage_log.

    pricing defaults to config.yaml's llm.pricing (input_per_1m,
    output_per_1m). This is an estimate, not a bill: it prices every call
    at the one model configured for this project, and misses anything a
    call didn't report (e.g. a step whose usage was never recorded).
    """
    if pricing is None:
        pricing = load_settings().get("pricing") or {}
    input_rate = pricing.get("input_per_1m", 0)
    output_rate = pricing.get("output_per_1m", 0)
    total = 0.0
    for call in usage_log or []:
        total += (call.get("input_tokens") or 0) / 1_000_000 * input_rate
        total += (call.get("output_tokens") or 0) / 1_000_000 * output_rate
    return round(total, 6)

# The exact shape the model must reply in. With "strict": True the API
# refuses to return anything that does not match (structured output).
ROUTING_SCHEMA = {
    "name": "routing_record",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["price", "policy", "both", "none"]},
            "check_premise": {"type": "boolean"},
            "answerable": {"type": "boolean"},
            "reason": {"type": "string"},
            "needs_clarification": {"type": "boolean"},
            "clarifying_question": {"type": ["string", "null"]},
        },
        "required": ["route", "check_premise", "answerable", "reason",
                     "needs_clarification", "clarifying_question"],
        "additionalProperties": False,
    },
}

# Separate schema/prompt keeps the baseline request unchanged for ablation.
DECOMPOSITION_SCHEMA = deepcopy(ROUTING_SCHEMA)
DECOMPOSITION_SCHEMA["name"] = "routing_tasks"
DECOMPOSITION_SCHEMA["schema"]["properties"]["tasks"] = {
    "type": "array", "maxItems": 6,
    "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "agent": {"type": "string", "enum": ["price", "policy"]},
            "question": {"type": "string"},
            "required_elements": {"type": "array", "minItems": 1,
                                  "items": {"type": "string"}},
        },
        "required": ["id", "agent", "question", "required_elements"],
    },
}
DECOMPOSITION_SCHEMA["schema"]["required"].append("tasks")


# Load the LLM configuration and check that a model has been specified.
# Return the settings used to build the routing request.
def load_settings():
    """Read the model settings from config.yaml (never typed into code)."""
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f) or {}
    llm = config.get("llm", {})
    if not llm.get("model"):
        raise ValueError("No model set. Add llm.model to config.yaml.")
    return llm


# Read the supervisor prompt template and replace its date placeholder
# with the supplied date string so the model has the correct date context.
def load_prompt(today):
    """Read the prompt file and put today's date into it."""
    template = PROMPT_FILE.read_text()
    return template.replace("{today}", today)


# Load environment variables from the project's .env file, then create a
# Groq client through its OpenAI-compatible endpoint.
def make_client(timeout=None):
    """Create the Groq-compatible OpenAI client using GROQ_API_KEY in .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Add GROQ_API_KEY to .env.")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=timeout)


# Classify a question by sending it and the dated supervisor prompt to the LLM.
# Use the supplied client and date, or create a client and use today's date.
# Request a response matching ROUTING_SCHEMA, include optional temperature
# and seed settings when configured, and parse the JSON into a dictionary
# containing route, check_premise, answerable, and reason.
def route_question(question, client=None, today=None, usage_log=None, decompose=False):
    """Classify one question. Returns a dict with route, check_premise,
    answerable and reason."""
    settings = load_settings()
    today = today or date.today().isoformat()
    client = client or make_client(settings.get("timeout_seconds"))

    prompt = load_prompt(today)
    if decompose:
        prompt += "\n\n" + (Path(__file__).parent / "prompts" / "decomposition_v1.md").read_text()
    request = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        "response_format": {"type": "json_schema", "json_schema":
                            DECOMPOSITION_SCHEMA if decompose else ROUTING_SCHEMA},
    }
    # Only send these if config.yaml sets them; some models reject them.
    if settings.get("temperature") is not None:
        request["temperature"] = settings["temperature"]
    if settings.get("seed") is not None:
        request["seed"] = settings["seed"]
    if settings.get("max_output_tokens") is not None:
        request["max_completion_tokens"] = settings["max_output_tokens"]

    response = client.chat.completions.create(**request)
    record_usage(usage_log, "router", response)
    message = response.choices[0].message
    if message.content is None:
        refusal = getattr(message, "refusal", None)
        raise RouterReplyError(f"No content in reply (refusal: {refusal})")
    return json.loads(message.content)


# Try the router when this file is run directly.
if __name__ == "__main__":
    import sys

    # Use the terminal question, or the example if none is provided.
    question = " ".join(sys.argv[1:]) or (
        "What was the median resale price of a 4-room flat in Tampines in 2024?"
    )
    # Print the routing decision as readable JSON.
    print(json.dumps(route_question(question), indent=2))
