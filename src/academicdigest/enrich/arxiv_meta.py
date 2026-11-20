"""arXiv API metadata: author comments ("Accepted at ICLR 2027, code at github...")
and the primary category, which the RSS feed does not carry. Also fills in title-less
mentions (an HN link to an older arXiv paper) with abstract, authors and subjects.

arXiv asks for at most one API request every 3 seconds.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import httpx

from ..db import ItemPatch
from ..models import Item
from ..pipeline.normalize import arxiv_id_from, normalize_doi
from ..sources.rss import parse_feed
from .base import Enricher, chunks, register

log = logging.getLogger(__name__)

API = "https://export.arxiv.org/api/query"


def parse_api(content: bytes) -> dict[str, dict]:
    """arXiv API Atom response -> {arxiv_id: metadata}."""
    out: dict[str, dict] = {}
    for e in parse_feed(content).entries:
        aid = arxiv_id_from(e.get("id"))
        if not aid:
            continue
        primary = (e.get("arxiv_primary_category") or {}).get("term")
        subjects = [t["term"] for t in e.get("tags", []) if t.get("term")]
        if primary:
            subjects = [primary] + [s for s in subjects if s != primary]
        out[aid] = {
            "comment": " ".join((e.get("arxiv_comment") or "").split()) or None,
            "journal_ref": e.get("arxiv_journal_ref") or None,
            "doi": normalize_doi(e.get("arxiv_doi")),
            "abstract": " ".join((e.get("summary") or "").split()) or None,
            "authors": [a["name"] for a in e.get("authors", []) if a.get("name")],
            "subjects": subjects,
        }
    return out


@register("arxiv_meta")
class ArxivMetaEnricher(Enricher):
    def wants(self, item: Item) -> bool:
        return bool(item.arxiv_id) and "arxiv_meta" not in item.extra

    def enrich(self, items: list[Item]) -> dict[int, ItemPatch]:
        by_id: dict[str, list[Item]] = defaultdict(list)
        for i in items[: int(self.opt("max_items", 2000))]:
            by_id[i.arxiv_id].append(i)  # type: ignore[index]
        delay = float(self.opt("delay_seconds", 3.0))
        patches: dict[int, ItemPatch] = {}
        for n, batch in enumerate(chunks(sorted(by_id), int(self.opt("batch_size", 100)))):
            if n:
                time.sleep(delay)
            try:
                resp = self.client.get(
                    self.opt("url", API),
                    params={"id_list": ",".join(batch), "max_results": len(batch)},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:  # retried on the next collect
                log.warning("arXiv API batch failed: %s", exc)
                continue
            meta = parse_api(resp.content)
            for aid in batch:
                m = meta.get(aid)
                if m is None:
                    continue
                extra = {"arxiv_meta": True, "primary_subject": m["subjects"][0] if m["subjects"] else None}
                if m["comment"]:
                    extra["comment"] = m["comment"]
                if m["journal_ref"]:
                    extra["journal_ref"] = m["journal_ref"]
                for item in by_id[aid]:
                    patches[item.id] = ItemPatch(
                        extra=extra,
                        excerpt=m["abstract"],
                        authors=m["authors"],
                        subjects=m["subjects"],
                        doi=m["doi"],
                    )
        return patches
