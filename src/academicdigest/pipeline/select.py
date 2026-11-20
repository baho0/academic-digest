"""Pick the papers that make it into the email."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from ..models import Cluster, Digest, DigestSection
from .dedupe import title_tokens

TOP_KEY = "top"


def already_sent(c: Cluster, sent_keys: set[str], sent_titles: list[str]) -> bool:
    keys = c.paper_keys | {f"url:{u}" for u in c.canonical_urls}
    if keys & sent_keys:
        return True
    norm = " ".join(title_tokens(c.title))
    return any(fuzz.token_sort_ratio(norm, " ".join(title_tokens(t))) >= 92 for t in sent_titles)


def select(
    clusters: list[Cluster],
    cfg: dict[str, Any],
    category_titles: dict[str, str],
    date: str,
    sent_keys: set[str] = frozenset(),  # type: ignore[assignment]
    sent_titles: list[str] = (),  # type: ignore[assignment]
) -> Digest:
    sel = cfg.get("selection", {})
    top_n = int(sel.get("top_n", 5))
    per_cat = int(sel.get("per_category", 5))
    min_score = float(sel.get("min_score", 4.0))
    max_total = int(sel.get("max_total", 25))
    cat_min = {k: float(v) for k, v in (sel.get("category_min_score") or {}).items()}

    eligible = sorted(
        (
            c
            for c in clusters
            if c.score >= cat_min.get(c.category, min_score)
            and not already_sent(c, sent_keys, list(sent_titles))
        ),
        key=lambda c: c.score,
        reverse=True,
    )

    top = eligible[:top_n]
    rest = eligible[top_n:]
    by_cat = {key: [c for c in rest if c.category == key][:per_cat] for key in category_titles}
    # Over max_total? Trim the weakest pick of the fullest section (highlights
    # stay), so a crowded field cannot push a quieter one out of the email.
    total = len(top) + sum(len(v) for v in by_cat.values())
    while total > max_total and any(by_cat.values()):
        fullest = max(by_cat, key=lambda k: (len(by_cat[k]), -by_cat[k][-1].score if by_cat[k] else 0))
        by_cat[fullest].pop()
        total -= 1

    sections = [DigestSection(TOP_KEY, str(sel.get("top_title", "Today's highlights")), top)] if top else []
    for key, title in category_titles.items():
        if by_cat[key]:
            sections.append(DigestSection(key, title, by_cat[key]))
    return Digest(date=date, sections=sections, candidates=len(clusters))
