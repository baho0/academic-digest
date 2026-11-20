"""Hacker News stories that link to papers (Algolia search restricted to URL domains).

A paper reaching the HN front page is a strong "people outside the field care"
signal; these items are mentions and merge with the arXiv/journal record.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from ..models import MENTION, RawItem
from ..pipeline.normalize import arxiv_id_from, clean_title, doi_from_url
from .base import Source, register

API = "https://hn.algolia.com/api/v1/search"
DOMAINS = ["arxiv.org", "openreview.net", "nature.com", "science.org", "biorxiv.org", "aclanthology.org"]


@register("hackernews")
class HackerNewsSource(Source):
    def fetch(self) -> list[RawItem]:
        since = int(time.time() - float(self.opt("lookback_hours", 48)) * 3600)
        min_points = int(self.opt("min_points", 30))
        hits: dict[str, dict] = {}
        for dom in self.opt("domains", DOMAINS):
            params = {
                "query": dom,
                "restrictSearchableAttributes": "url",
                "tags": "story",
                "numericFilters": f"created_at_i>{since},points>{min_points}",
                "hitsPerPage": int(self.opt("max_items", 50)),
            }
            for h in self.get(self.config.url or API, params=params).json().get("hits", []):
                if dom in (h.get("url") or ""):
                    hits[str(h["objectID"])] = h

        items: list[RawItem] = []
        for oid, h in hits.items():
            url = h["url"]
            hn_url = f"https://news.ycombinator.com/item?id={oid}"
            items.append(
                self.item(
                    guid=oid,
                    kind=MENTION,
                    title=clean_title(h.get("title", "")),
                    url=url,
                    arxiv_id=arxiv_id_from(url),
                    doi=doi_from_url(url),
                    published_at=datetime.fromtimestamp(h["created_at_i"], UTC),
                    discussion_url=hn_url,
                    metrics={
                        "hn_points": float(h.get("points") or 0),
                        "hn_comments": float(h.get("num_comments") or 0),
                    },
                )
            )
        return items
