"""Summaries for selected papers.

`Summarizer` is the extension point (`summarizer:` in scoring.yaml):
  excerpt  the opening sentences of the abstract
  llm      a short summary written by the configured LLM; falls back to the
           excerpt for any paper the LLM fails on
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..llm import LLMBudgetExceeded, LLMClient, LLMError, LLMRequest, render_prompt
from ..models import Cluster
from .normalize import truncate

log = logging.getLogger(__name__)


class Summarizer(Protocol):
    def summarize(self, clusters: list[Cluster]) -> None:
        """Set `summary` on every cluster (in place)."""


class ExcerptSummarizer:
    def __init__(self, max_chars: int = 350) -> None:
        self.max_chars = max_chars

    def summarize(self, clusters: list[Cluster]) -> None:
        for c in clusters:
            text = c.abstract
            c.summary = truncate(text, self.max_chars) if text and text.strip() != c.title.strip() else None


class LLMSummarizer:
    def __init__(self, llm: LLMClient, max_chars: int = 400, language: str = "English") -> None:
        self.llm = llm
        self.max_chars = max_chars
        self.language = language
        self.fallback = ExcerptSummarizer(max_chars)

    def summarize(self, clusters: list[Cluster]) -> None:
        budget_hit = False
        for c in clusters:
            if not budget_hit and c.abstract:
                system, prompt = render_prompt(
                    "summarize", paper=c, max_chars=self.max_chars, language=self.language
                )
                try:
                    text = self.llm.complete(
                        LLMRequest(prompt=prompt, system=system, max_tokens=400, purpose="summary")
                    ).text.strip()
                    if text:
                        c.summary = truncate(text, self.max_chars + 100)
                        continue
                except LLMBudgetExceeded as exc:
                    log.warning("%s; using abstract excerpts for the rest", exc)
                    budget_hit = True
                except LLMError as exc:
                    log.warning("LLM summary failed for %r: %s", c.title[:60], exc)
            self.fallback.summarize([c])


def build_summarizer(cfg: dict[str, Any], llm: LLMClient | None = None) -> Summarizer:
    engine = cfg.get("summarizer", "excerpt")
    llm_cfg = cfg.get("llm", {}) or {}
    max_chars = int(llm_cfg.get("summary_max_chars", 400))
    if engine == "excerpt":
        return ExcerptSummarizer(int(cfg.get("excerpt_max_chars", 350)))
    if engine == "llm":
        if llm is None:
            raise ValueError("summarizer: llm needs an LLM provider (AD_LLM_PROVIDER)")
        return LLMSummarizer(llm, max_chars, str(llm_cfg.get("language", "English")))
    raise ValueError(f"unknown summarizer {engine!r} (excerpt | llm)")
