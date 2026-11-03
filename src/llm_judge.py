import yaml
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


JUDGE_RUBRIC = {
    "evidence_support": {
        "description": (
            "Are the answer's factual claims supported "
            "by the retrieved evidence?"
        ),
        "score_range": [0, 2],
    },
    "policy_currency": {
        "description": (
            "Does the answer use the correct applicable "
            "policy version based on the available dates?"
        ),
        "score_range": [0, 2],
    },
    "citation_relevance": {
        "description": (
            "Do the retrieved passages directly support "
            "the answer?"
        ),
        "score_range": [0, 2],
    },
    "unsupported_claims": {
        "description": (
            "Does the answer avoid introducing information "
            "not supported by the evidence?"
        ),
        "score_range": [0, 2],
    },
    "appropriate_abstention": {
        "description": (
            "When evidence is insufficient, does the answer "
            "avoid guessing and abstain appropriately?"
        ),
        "score_range": [0, 2],
    },
}


def max_score():
    return sum(
        criterion["score_range"][1]
        for criterion in JUDGE_RUBRIC.values()
    )


def build_judge_prompt(
    question,
    answer,
    evidence,
):
    evidence_text = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        evidence_text.append(
            f"""
[Evidence {index}]
Title: {item.get("title")}
Source: {item.get("source")}
Effective date: {item.get("effective_date")}
Text: {item.get("text")}
""".strip()
        )

    rubric_text = []

    for name, criterion in JUDGE_RUBRIC.items():
        rubric_text.append(
            f"- {name}: {criterion['description']} "
            "(0 = poor, 1 = partial, 2 = fully satisfactory)"
        )

    return f"""
Evaluate the answer strictly using only the supplied evidence.

Question:
{question}

Answer:
{answer}

Evidence:
{chr(10).join(evidence_text)}

Rubric:
{chr(10).join(rubric_text)}

Return JSON only in this format:

{{
  "evidence_support": 0,
  "policy_currency": 0,
  "citation_relevance": 0,
  "unsupported_claims": 0,
  "appropriate_abstention": 0,
  "total_score": 0,
  "reason": "Brief explanation"
}}
""".strip()


def validate_judge_output(raw_output):
    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Judge output was not valid JSON."
        ) from exc

    score_fields = [
        "evidence_support",
        "policy_currency",
        "citation_relevance",
        "unsupported_claims",
        "appropriate_abstention",
    ]

    for field in score_fields:
        if field not in result:
            raise ValueError(
                f"Missing judge field: {field}"
            )

        if result[field] not in {0, 1, 2}:
            raise ValueError(
                f"Invalid score for {field}: "
                f"{result[field]}"
            )

    expected_total = sum(
        result[field]
        for field in score_fields
    )

    result["total_score"] = expected_total

    if not isinstance(
        result.get("reason"),
        str,
    ):
        raise ValueError(
            "Judge reason must be a string."
        )

    return result


ENV_PATH = (
    Path(__file__).resolve().parents[1]
    / ".env"
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config.yaml"
)


 def_placeholder
