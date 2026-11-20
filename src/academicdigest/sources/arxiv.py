"""arXiv daily announcements via rss.arxiv.org (one feed for many categories).

Only `new` and `cross` (cross-listed) announcements are kept: `replace` is a
new version of an old paper, not news. Comments ("Accepted at NeurIPS...")
are not in the feed; the `arxiv_meta` enricher adds them from the arXiv API.
"""

from __future__ import annotations

import re

from ..models import RawItem
from ..pipeline.normalize import arxiv_id_from, clean_title, normalize_doi, split_authors
from .base import Source, register, struct_to_dt
from .rss import parse_feed

FEED = "https://rss.arxiv.org/atom/{categories}"
_PREAMBLE = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\S+\s*(Abstract:\s*)?", re.I)


def clean_abstract(summary: str) -> str:
    return " ".join(_PREAMBLE.sub("", summary or "").split())


@register("arxiv")
class ArxivSource(Source):
    def fetch(self) -> list[RawItem]:
        categories = self.opt("categories", ["cs.AI"])
        url = self.config.url or FEED.format(categories="+".join(categories))
        keep = set(self.opt("announce_types", ["new", "cross"]))
        feed = parse_feed(self.get(url).content)
        items: list[RawItem] = []
        for e in feed.entries[: int(self.opt("max_items", 2000))]:
            kind = e.get("arxiv_announce_type", "new")
            aid = arxiv_id_from(e.get("link")) or arxiv_id_from(e.get("id", "").split(":")[-1])
            if kind not in keep or not aid:
                continue
            extra = {"announce_type": kind}
            if e.get("arxiv_journal_reference"):
                extra["journal_ref"] = e["arxiv_journal_reference"]
            items.append(
                self.item(
                    guid=aid,
                    title=clean_title(e.get("title", "")),
                    url=f"https://arxiv.org/abs/{aid}",
                    excerpt=clean_abstract(e.get("summary", "")) or None,
                    authors=split_authors(e.get("author")),
                    subjects=[t["term"] for t in e.get("tags", []) if t.get("term")],
                    arxiv_id=aid,
                    doi=normalize_doi(e.get("arxiv_doi")),
                    published_at=struct_to_dt(e.get("published_parsed")),
                    extra=extra,
                )
            )
        return items
