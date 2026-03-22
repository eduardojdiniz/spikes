#!/usr/bin/env python3
"""CLI entry point for the Contrastive Refinement Engine.

Usage:
    python main.py "Your base query here"
    python main.py --iterations 3 --eval-mode llm "Your query"
    python main.py --provider openai --model gpt-4o "Your query"
    python main.py --eval-mode interactive "Your query"
    python main.py --eval-mode random "Your query"   # for testing
"""

from __future__ import annotations

import argparse
import sys

from orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrastive Refinement Engine — iteratively refine "
        "LLM outputs via contrastive feedback loops.",
    )
    parser.add_argument(
        "query",
        help="The base query to generate outputs for.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum number of refinement iterations (default: 5).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["llm", "interactive", "random"],
        default="llm",
        help="Evaluation strategy: 'llm' (auto), 'interactive' (human), "
        "'random' (test baseline). Default: llm.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider (default: anthropic).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for generation (default: provider-specific).",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="Model for evaluation (default: same as --model).",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Path to export the full run as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )

    args = parser.parse_args()

    orchestrator = Orchestrator(
        query=args.query,
        max_iterations=args.iterations,
        eval_mode=args.eval_mode,
        provider=args.provider,
        model=args.model,
        eval_model=args.eval_model,
        verbose=not args.quiet,
    )

    orchestrator.execute()

    if args.export:
        orchestrator.export(args.export)


if __name__ == "__main__":
    main()
