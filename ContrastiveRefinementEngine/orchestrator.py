"""Orchestrator for the Contrastive Refinement Engine.

Drives the full loop: generate -> evaluate -> feed back -> refine -> repeat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from engine import ContrastiveEngine, GenerationResult
from evaluator import (
    EvaluationResult,
    evaluate_interactive,
    evaluate_random,
    evaluate_with_llm,
)


@dataclass
class RefinementRun:
    """Complete record of a refinement session."""

    query: str
    iterations: list[dict] = field(default_factory=list)


class Orchestrator:
    """Runs the contrastive refinement loop for a configurable number of iterations."""

    def __init__(
        self,
        query: str,
        max_iterations: int = 5,
        eval_mode: str = "llm",
        provider: str = "anthropic",
        model: str | None = None,
        eval_model: str | None = None,
        api_key: str | None = None,
        verbose: bool = True,
    ):
        self.query = query
        self.max_iterations = max_iterations
        self.eval_mode = eval_mode
        self.provider = provider
        self.model = model
        self.eval_model = eval_model
        self.api_key = api_key
        self.verbose = verbose

        self.engine = ContrastiveEngine(
            provider=provider, model=model, api_key=api_key
        )
        self.run = RefinementRun(query=query)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _evaluate(self, items: list[str]) -> EvaluationResult:
        if self.eval_mode == "llm":
            return evaluate_with_llm(
                query=self.query,
                items=items,
                provider=self.provider,
                model=self.eval_model or self.model,
                api_key=self.api_key,
            )
        elif self.eval_mode == "interactive":
            return evaluate_interactive(items)
        else:
            return evaluate_random(items)

    def _print_items(self, result: GenerationResult) -> None:
        self._log(f"\n{'='*60}")
        self._log(f"  TURN {result.turn} — Generated {len(result.items)} items")
        self._log(f"{'='*60}")
        for i, item in enumerate(result.items):
            self._log(f"  {i+1}. {item}")

        if result.contrastive_analysis:
            self._log(f"\n--- Contrastive Analysis ---")
            self._log(result.contrastive_analysis)
        if result.synthesized_rules:
            self._log(f"\n--- Synthesized Rules ---")
            self._log(result.synthesized_rules)

    def _print_evaluation(self, evaluation: EvaluationResult) -> None:
        self._log(f"\n--- Evaluation ---")
        self._log(f"  Positives: {evaluation.positives}")
        self._log(f"  Negatives: {evaluation.negatives}")
        if evaluation.reasoning:
            self._log(f"  Reasoning: {evaluation.reasoning}")

    def execute(self) -> RefinementRun:
        """Run the full contrastive refinement loop."""
        # Turn 1: Initialize
        self._log(f"\n[Orchestrator] Starting contrastive refinement loop")
        self._log(f"[Orchestrator] Query: {self.query}")
        self._log(f"[Orchestrator] Max iterations: {self.max_iterations}")
        self._log(f"[Orchestrator] Eval mode: {self.eval_mode}")

        result = self.engine.initialize(self.query)
        self._print_items(result)

        iteration_record = {
            "turn": 1,
            "items": result.items,
            "raw_response": result.raw_response,
        }
        self.run.iterations.append(iteration_record)

        # Turns 2..N: Evaluate + Refine
        for i in range(self.max_iterations - 1):
            self._log(f"\n[Orchestrator] Evaluating turn {result.turn} outputs...")
            evaluation = self._evaluate(result.items)
            self._print_evaluation(evaluation)

            if not evaluation.positives or not evaluation.negatives:
                self._log(
                    "[Orchestrator] All items classified the same — "
                    "stopping early (no contrastive signal)."
                )
                break

            self._log(f"\n[Orchestrator] Refining (turn {result.turn + 1})...")
            result = self.engine.refine(
                evaluation.positives, evaluation.negatives
            )
            self._print_items(result)

            iteration_record = {
                "turn": result.turn,
                "items": result.items,
                "positives_from_prev": evaluation.positives,
                "negatives_from_prev": evaluation.negatives,
                "contrastive_analysis": result.contrastive_analysis,
                "synthesized_rules": result.synthesized_rules,
                "raw_response": result.raw_response,
            }
            self.run.iterations.append(iteration_record)

        self._log(f"\n[Orchestrator] Refinement loop complete after {result.turn} turns.")
        return self.run

    def export(self, path: str | None = None) -> str:
        """Export the full run as JSON."""
        data = {
            "query": self.run.query,
            "total_turns": len(self.run.iterations),
            "iterations": self.run.iterations,
            "engine_history": json.loads(self.engine.export_history()),
        }
        output = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
            self._log(f"[Orchestrator] Exported to {path}")
        return output
