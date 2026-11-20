"""Semantic Scholar: citation counts and the highest author h-index.

Semantic Scholar usually indexes a new arXiv paper a day or two after it is
announced, so papers not found yet are retried after `refresh_hours`.
Unauthenticated requests share a global rate limit; set AD_S2_API_KEY to get
your own (https://www.semanticscholar.org/product/api).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

import httpx

from ..db import ItemPatch
from ..models import Item
from ..sources.base import parse_iso
from .base import Enricher, chunks, register

log = logging.getLogger(__name__)

API = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "citationCount,influentialCitationCount,authors.hIndex"


def s2_id(item: Item) -> str | None:
    if item.arxiv_id:
        return f"ARXIV:{item.arxiv_id}"
    if item.doi:
        return f"DOI:{item.doi}"
    return None


@register("semantic_scholar")
class SemanticScholarEnricher(Enricher):
    def wants(self, item: Item) -> bool:
        if not s2_id(item):
            return False
        checked = parse_iso(item.extra.get("s2_checked_at"))
        if checked is None:
            return True
        found = "s2_citations" in item.metrics
        refresh = float(self.opt("refresh_hours", 12 if not found else 72))
        return datetime.now(UTC) - checked > timedelta(hours=refresh)

    def enrich(self, items: list[Item]) -> dict[int, ItemPatch]:
        headers = {"x-api-key": self.settings.s2_api_key} if self.settings.s2_api_key else {}
        now = datetime.now(UTC).isoformat()
        by_id: dict[str, list[Item]] = {}
        for i in items[: int(self.opt("max_items", 1000))]:
            by_id.setdefault(s2_id(i), []).append(i)  # type: ignore[arg-type]
        patches: dict[int, ItemPatch] = {}
        for n, batch in enumerate(chunks(list(by_id), int(self.opt("batch_size", 500)))):
            if n:
                time.sleep(float(self.opt("delay_seconds", 1.5)))
            try:
                resp = self.client.post(
                    self.opt("url", API), params={"fields": FIELDS}, json={"ids": batch}, headers=headers
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Semantic Scholar batch failed: %s", exc)
                continue
            for key, paper in zip(batch, resp.json()):
                metrics: dict[str, float] = {}
                if paper:
                    hidx = [a.get("hIndex") or 0 for a in paper.get("authors") or []]
                    metrics = {
                        "s2_citations": float(paper.get("citationCount") or 0),
                        "s2_influential": float(paper.get("influentialCitationCount") or 0),
                        "s2_max_hindex": float(max(hidx, default=0)),
                    }
                for item in by_id[key]:
                    patches[item.id] = ItemPatch(metrics=metrics, extra={"s2_checked_at": now})
        return patches
