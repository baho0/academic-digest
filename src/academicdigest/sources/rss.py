"""Journal / generic RSS & Atom feeds (Nature, Nature Machine Intelligence, lab blogs...).

Options:
  url_include   regex; keep only entries whose link matches (e.g. Nature research
                articles are /articles/s41586-..., news pieces are /articles/d41586-...)
  max_items     default 50
"""

from __future__ import annotations

import re

import feedparser

from ..models import RawItem
from ..pipeline.normalize import clean_title, doi_from_url, normalize_doi, strip_html, truncate
from .base import Source, register, struct_to_dt

# "Nature, Published online: 19 November 2026; doi:10.1038/s41586-026-11445-z"
_PUBLISHER_PREAMBLE = re.compile(r"^[^;]{0,150}Published online:[^;]*;\s*doi:\S+\s*", re.I)


def parse_feed(content: bytes):
    return feedparser.parse(content)


def entry_doi(e) -> str | None:
    for key in ("prism_doi", "dc_identifier", "id"):
        doi = normalize_doi(e.get(key))
        if doi:
            return doi
    return doi_from_url(e.get("link"))


@register("journal")
@register("rss")
class FeedSource(Source):
    def fetch(self) -> list[RawItem]:
        feed = parse_feed(self.get(self.config.url).content)
        include = self.opt("url_include")
        include_re = re.compile(include) if include else None
        items: list[RawItem] = []
        for e in feed.entries[: int(self.opt("max_items", 50))]:
            link = e.get("link")
            title = clean_title(e.get("title", ""))
            if not link or not title or (include_re and not include_re.search(link)):
                continue
            summary = e.get("summary") or ""
            if not summary and e.get("content"):
                summary = e.content[0].get("value", "")
            extra = {}
            if e.get("prism_publicationname"):
                extra["journal"] = e["prism_publicationname"]
            items.append(
                self.item(
                    guid=e.get("id") or link,
                    title=title,
                    url=link,
                    excerpt=truncate(_PUBLISHER_PREAMBLE.sub("", strip_html(summary)), 1500) or None,
                    authors=[a["name"] for a in e.get("authors", []) if a.get("name")],
                    doi=entry_doi(e),
                    published_at=struct_to_dt(e.get("published_parsed") or e.get("updated_parsed")),
                    extra=extra,
                )
            )
        return items
