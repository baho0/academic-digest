"""LLM infrastructure: provider-agnostic client, response cache, budget and prompts.

Usage from the pipeline:

    llm = build_llm(settings, session)     # None when AD_LLM_PROVIDER=none
    if llm:
        system, prompt = render_prompt("summarize", paper=cluster, max_chars=400, language="English")
        text = llm.complete(LLMRequest(prompt=prompt, system=system, purpose="summary")).text
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BudgetedLLM,
    LLMBudgetExceeded,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    build_provider,
    providers,
    register_provider,
)
from .cache import CachedLLM
from .prompts import parse_json, render_prompt

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ..settings import Settings


def build_llm(settings: Settings, session: Session | None = None) -> LLMClient | None:
    """Configured provider behind the per-run call budget and (with a session) the DB cache."""
    provider = build_provider(settings)
    if provider is None:
        return None
    client: LLMClient = BudgetedLLM(provider, settings.llm_max_calls)
    if session is not None:
        client = CachedLLM(client, session)
    return client


__all__ = [
    "BudgetedLLM",
    "CachedLLM",
    "LLMBudgetExceeded",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "build_llm",
    "parse_json",
    "providers",
    "register_provider",
    "render_prompt",
]
