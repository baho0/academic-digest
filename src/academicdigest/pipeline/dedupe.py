"""Group items about the same paper into clusters.

Merge rules (union-find):
  1. same arXiv id (any version, abs/pdf/html links) or same DOI
  2. same canonical URL
  3. very similar titles from *different* sources (journal version of a
     preprint, an HN title copied from the paper) — but never two items that
     both carry an arXiv id: different ids are different papers.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from rapidfuzz import fuzz

from ..models import Cluster, Item

_PUNCT = re.compile(r"[^\w\s.]")
_STOP = frozenset(
    "a an the and or of to in on for with by at from as is are was were be its it this that "
    "via towards toward using paper".split()
)


def title_tokens(title: str) -> list[str]:
    return [t for t in _PUNCT.sub(" ", title.lower()).split() if t not in _STOP]


def item_keys(item: Item) -> list[str]:
    keys = []
    if item.arxiv_id:
        keys.append(f"arxiv:{item.arxiv_id}")
    if item.doi:
        keys.append(f"doi:{item.doi}")
        if item.doi.startswith("10.48550/arxiv."):
            keys.append(f"arxiv:{item.doi.removeprefix('10.48550/arxiv.')}")
    if item.canonical_url:
        keys.append(f"url:{item.canonical_url}")
    return keys


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_items(items: list[Item], cfg: dict[str, Any] | None = None) -> list[Cluster]:
    cfg = cfg or {}
    threshold = float(cfg.get("title_similarity", 92))
    min_tokens = int(cfg.get("min_title_tokens", 5))

    uf = _UnionFind(len(items))

    by_key: dict[str, int] = {}
    for idx, item in enumerate(items):
        for key in item_keys(item):
            if key in by_key:
                uf.union(by_key[key], idx)
            else:
                by_key[key] = idx

    # Fuzzy titles: only items without an arXiv id can be the "other side"
    # (journal articles, mentions linking elsewhere), which keeps this cheap
    # even with a thousand arXiv papers in the window.
    norm = [" ".join(title_tokens(i.title)) for i in items]
    long_enough = [len(n.split()) >= min_tokens for n in norm]
    loose = [i for i, it in enumerate(items) if not it.arxiv_id and long_enough[i]]
    loose_set = set(loose)
    for a in loose:
        for b in range(len(items)):
            if b == a or not long_enough[b] or items[a].source == items[b].source:
                continue
            if b in loose_set and b < a:  # pair already compared from the other side
                continue
            if uf.find(a) == uf.find(b):
                continue
            if fuzz.token_sort_ratio(norm[a], norm[b]) >= threshold:
                uf.union(a, b)

    groups: dict[int, list[Item]] = defaultdict(list)
    for idx, item in enumerate(items):
        groups[uf.find(idx)].append(item)
    return [Cluster(items=g) for g in groups.values()]
