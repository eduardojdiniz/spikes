"""LLM engine wrapper for the Contrastive Refinement Engine.

Supports Anthropic Claude and OpenAI-compatible APIs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from system_prompt import SYSTEM_PROMPT, format_base_query, format_feedback


@dataclass
class GenerationResult:
    """Result from a single generation turn."""

    turn: int
    raw_response: str
    items: list[str]
    contrastive_analysis: str | None = None
    synthesized_rules: str | None = None


@dataclass
class ConversationState:
    """Tracks the multi-turn conversation with the LLM."""

    messages: list[dict[str, str]] = field(default_factory=list)
    turn: int = 0
    results: list[GenerationResult] = field(default_factory=list)


def _parse_numbered_items(text: str) -> list[str]:
    """Extract numbered items (1. ... 2. ... etc.) from LLM output."""
    pattern = r"(?:^|\n)\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.\s+|\n###|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    items = []
    for _num, content in matches:
        cleaned = content.strip()
        if cleaned:
            items.append(cleaned)
    return items


def _parse_contrastive_sections(text: str) -> tuple[str | None, str | None]:
    """Extract contrastive analysis and synthesized rules sections."""
    analysis = None
    rules = None

    analysis_match = re.search(
        r"###\s*1\.\s*CONTRASTIVE ANALYSIS\s*\n(.*?)(?=\n###|\Z)",
        text,
        re.DOTALL,
    )
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    rules_match = re.search(
        r"###\s*2\.\s*SYNTHESIZED GENERATION RULES\s*\n(.*?)(?=\n###|\Z)",
        text,
        re.DOTALL,
    )
    if rules_match:
        rules = rules_match.group(1).strip()

    return analysis, rules


class ContrastiveEngine:
    """Drives the contrastive refinement loop via an LLM API."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model or self._default_model()
        self.api_key = api_key or self._resolve_api_key()
        self.state = ConversationState()

    def _default_model(self) -> str:
        if self.provider == "anthropic":
            return "claude-sonnet-4-20250514"
        return "gpt-4o"

    def _resolve_api_key(self) -> str:
        if self.provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            key = os.environ.get("OPENAI_API_KEY", "")
        return key

    def _call_anthropic(self, messages: list[dict]) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    def _call_openai(self, messages: list[dict]) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=sys_msg + messages,
        )
        return response.choices[0].message.content

    def _call_llm(self, messages: list[dict]) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai(messages)

    def initialize(self, base_query: str) -> GenerationResult:
        """Turn 1: Send the base query and get the initial 10 outputs."""
        user_msg = format_base_query(base_query)
        self.state.messages.append({"role": "user", "content": user_msg})
        self.state.turn = 1

        raw = self._call_llm(self.state.messages)
        self.state.messages.append({"role": "assistant", "content": raw})

        items = _parse_numbered_items(raw)
        result = GenerationResult(turn=1, raw_response=raw, items=items)
        self.state.results.append(result)
        return result

    def refine(
        self, positives: list[int], negatives: list[int]
    ) -> GenerationResult:
        """Turn 2+: Send contrastive feedback and get refined outputs."""
        self.state.turn += 1
        feedback_msg = format_feedback(positives, negatives)
        self.state.messages.append({"role": "user", "content": feedback_msg})

        raw = self._call_llm(self.state.messages)
        self.state.messages.append({"role": "assistant", "content": raw})

        items = _parse_numbered_items(raw)
        analysis, rules = _parse_contrastive_sections(raw)
        result = GenerationResult(
            turn=self.state.turn,
            raw_response=raw,
            items=items,
            contrastive_analysis=analysis,
            synthesized_rules=rules,
        )
        self.state.results.append(result)
        return result

    def export_history(self) -> str:
        """Export the full conversation and parsed results as JSON."""
        data = {
            "turns": self.state.turn,
            "messages": self.state.messages,
            "results": [
                {
                    "turn": r.turn,
                    "items": r.items,
                    "contrastive_analysis": r.contrastive_analysis,
                    "synthesized_rules": r.synthesized_rules,
                }
                for r in self.state.results
            ],
        }
        return json.dumps(data, indent=2)
