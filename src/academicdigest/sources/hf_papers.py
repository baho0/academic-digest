"""Hugging Face Daily Papers: community-curated ML papers with upvotes.

Upvotes keep growing during the day, so the last `days` lists are re-fetched
on every collect and the metric is refreshed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..models import RawItem
from ..pipeline.normalize import clean_title
from .base import Source, parse_iso, register

API = "https://huggingface.co/api/daily_papers"


@register("hf_papers")
class HfPapersSource(Source):
    def fetch(self) -> list[RawItem]:
        min_up = int(self.opt("min_upvotes", 3))
        today = datetime.now(UTC).date()
        papers: dict[str, dict] = {}
        for back in range(int(self.opt("days", 2))):
            day = (today - timedelta(days=back)).isoformat()
            data = self.get(self.config.url or API, params={"date": day, "limit": 100}).json()
            for p in data:
                papers.setdefault(p["paper"]["id"], p)

        items: list[RawItem] = []
        for pid, p in papers.items():
            paper = p["paper"]
            upvotes = int(paper.get("upvotes", 0))
            if upvotes < min_up:
                continue
            extra = {}
            if paper.get("githubRepo"):
                extra["code_url"] = paper["githubRepo"]
            if paper.get("ai_summary"):
                extra["hf_ai_summary"] = paper["ai_summary"]
            items.append(
                self.item(
                    guid=pid,
                    title=clean_title(p.get("title") or paper.get("title", "")),
                    url=f"https://arxiv.org/abs/{pid}",
                    excerpt=" ".join((p.get("summary") or paper.get("summary") or "").split()) or None,
                    authors=[a["name"] for a in paper.get("authors", []) if a.get("name")],
                    arxiv_id=pid,
                    published_at=parse_iso(paper.get("submittedOnDailyAt") or p.get("publishedAt")),
                    discussion_url=f"https://huggingface.co/papers/{pid}",
                    metrics={
                        "hf_upvotes": float(upvotes),
                        "hf_comments": float(p.get("numComments") or 0),
                    },
                    extra=extra,
                )
            )
        return items
