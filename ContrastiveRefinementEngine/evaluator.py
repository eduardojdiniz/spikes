"""Evaluator module for classifying generated outputs as Positive or Negative.

Supports three evaluation strategies:
1. LLM-based automatic evaluation (uses a second LLM call to judge quality)
2. Interactive human evaluation (CLI prompts)
3. Random baseline (for testing the loop without human/LLM input)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Classification of a generation batch into positive and negative items."""

    positives: list[int]  # 1-indexed item numbers
    negatives: list[int]  # 1-indexed item numbers
    reasoning: str | None = None


EVALUATOR_PROMPT = """\
You are a strict quality evaluator. You will receive a base query and a list \
of 10 generated responses. Your task is to classify each response as either \
POSITIVE (high quality, insightful, non-obvious, surprising) or NEGATIVE \
(generic, clichéd, already well-known, or low quality).

For context, the base query was:
{query}

Here are the 10 generated responses:
{items_text}

Respond with exactly this JSON format and nothing else:
{{
  "positives": [list of item numbers that are high quality],
  "negatives": [list of item numbers that are low quality],
  "reasoning": "Brief explanation of your classification criteria"
}}

Be discriminating. Classify roughly 40-60% as positive and the rest as \
negative. Every item must appear in exactly one list.
"""


def evaluate_with_llm(
    query: str,
    items: list[str],
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
) -> EvaluationResult:
    """Use an LLM to automatically classify items as positive/negative."""
    import json as json_mod

    items_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    prompt = EVALUATOR_PROMPT.format(query=query, items_text=items_text)

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        response = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
    else:
        import openai

        client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )
        response = client.chat.completions.create(
            model=model or "gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content

    # Parse JSON from LLM response
    # Handle potential markdown code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]

    data = json_mod.loads(cleaned)
    return EvaluationResult(
        positives=data["positives"],
        negatives=data["negatives"],
        reasoning=data.get("reasoning"),
    )


def evaluate_interactive(items: list[str]) -> EvaluationResult:
    """Prompt the user in the CLI to classify each item."""
    positives = []
    negatives = []

    print("\n--- EVALUATION ---")
    print("For each item, enter 'p' (positive) or 'n' (negative):\n")

    for i, item in enumerate(items):
        num = i + 1
        print(f"  {num}. {item}")
        while True:
            choice = input(f"  [{num}] p/n? ").strip().lower()
            if choice in ("p", "n"):
                break
            print("    Please enter 'p' or 'n'.")
        if choice == "p":
            positives.append(num)
        else:
            negatives.append(num)

    return EvaluationResult(
        positives=positives,
        negatives=negatives,
        reasoning="Human evaluation",
    )


def evaluate_random(items: list[str], positive_rate: float = 0.5) -> EvaluationResult:
    """Random baseline evaluator for testing the loop."""
    all_nums = list(range(1, len(items) + 1))
    random.shuffle(all_nums)
    split = max(1, int(len(all_nums) * positive_rate))
    positives = sorted(all_nums[:split])
    negatives = sorted(all_nums[split:])
    return EvaluationResult(
        positives=positives,
        negatives=negatives,
        reasoning="Random baseline classification",
    )
