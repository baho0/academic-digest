"""Prompt templates (llm/prompts/*.j2) and output parsing helpers.

Prompts live in files so they can be tuned without touching pipeline code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .base import LLMError

_DIR = Path(__file__).parent / "prompts"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render_prompt(name: str, **ctx: Any) -> tuple[str | None, str]:
    """Render `<name>.j2`; returns (system, prompt).

    The template's `system` block (if any) becomes the system prompt and the
    rest becomes the user prompt.
    """
    template = _env().get_template(f"{name}.j2")
    context = template.new_context(ctx)
    system = None
    if "system" in template.blocks:
        system = "".join(template.blocks["system"](context)).strip() or None
    prompt = "".join(template.blocks["prompt"](context)).strip()
    return system, prompt


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and chatter around it."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"no JSON object in model output: {text[:200]!r}") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON in model output: {exc}") from None
    if not isinstance(value, dict):
        raise LLMError("model output is not a JSON object")
    return value
