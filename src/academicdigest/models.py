"""Domain models shared by every pipeline stage (independent of storage)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

PAPER = "paper"      # a record of the paper itself (arXiv, journal, HF Papers)
MENTION = "mention"  # a discussion that links to a paper (Hacker News)

_GITHUB = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+", re.I)
_VENUE_HINT = re.compile(
    r"\b(accepted|to appear|published|proceedings|conference|workshop|journal|"
    r"NeurIPS|ICML|ICLR|ACL|EMNLP|NAACL|CVPR|ICCV|ECCV|AAAI|IJCAI|KDD|"
    r"USENIX|CCS|NDSS|OSDI|SOSP|SIGMOD|VLDB|CHI|STOC|FOCS|SODA)\b",
    re.I,
)


@dataclass
class RawItem:
    """One paper (or mention of a paper) as returned by a source, before storage."""

    source: str
    guid: str
    title: str
    url: str
    kind: str = PAPER
    tier: int = 2
    category_hint: str | None = None
    excerpt: str | None = None  # abstract for papers
    authors: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)  # arXiv categories, primary first
    arxiv_id: str | None = None  # without version: 2611.24943
    doi: str | None = None  # lowercase, no resolver prefix
    published_at: datetime | None = None
    discussion_url: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Item(RawItem):
    """A stored item (RawItem + storage fields)."""

    id: int = 0
    canonical_url: str = ""
    fetched_at: datetime | None = None

    @property
    def timestamp(self) -> datetime | None:
        return self.published_at or self.fetched_at


@dataclass
class ScoreLine:
    label: str
    delta: float

    def __str__(self) -> str:
        return f"{self.label} {self.delta:+.1f}"


@dataclass
class Cluster:
    """One paper, possibly seen through several sources (arXiv + HF + HN...)."""

    items: list[Item]
    category: str = ""
    topic_hits: float = 0.0  # evidence that this is in one of our fields at all
    score: float = 0.0
    breakdown: list[ScoreLine] = field(default_factory=list)
    summary: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)  # e.g. {"llm_reason": "..."}

    @property
    def lead(self) -> Item:
        """The item whose title/link represent the cluster: the paper record itself."""
        return min(
            self.items,
            key=lambda i: (
                0 if i.kind == PAPER else 1,
                0 if i.excerpt else 1,
                0 if i.authors else 1,
                i.tier,
            ),
        )

    @property
    def title(self) -> str:
        return self.lead.title

    @property
    def url(self) -> str:
        return self.lead.url

    @property
    def abstract(self) -> str:
        return next((i.excerpt for i in self._papers_first if i.excerpt), "") or ""

    @property
    def authors(self) -> list[str]:
        return next((i.authors for i in self._papers_first if i.authors), [])

    @property
    def subjects(self) -> list[str]:
        return next((i.subjects for i in self._papers_first if i.subjects), [])

    @property
    def arxiv_id(self) -> str | None:
        return next((i.arxiv_id for i in self.items if i.arxiv_id), None)

    @property
    def doi(self) -> str | None:
        return next((i.doi for i in self._papers_first if i.doi), None)

    @property
    def pdf_url(self) -> str | None:
        if self.arxiv_id:
            return f"https://arxiv.org/pdf/{self.arxiv_id}"
        return self.extra("pdf_url")

    @property
    def venue(self) -> str | None:
        """Where it is published / accepted: journal ref, journal name, or an arXiv
        comment that names a venue ("Accepted at ICLR 2027"; not "12 pages, 5 figures")."""
        found = self.extra("journal_ref") or self.extra("journal")
        if not found:
            comment = self.extra("comment") or ""
            found = comment if _VENUE_HINT.search(comment) else None
        return (found[:157] + "…") if found and len(found) > 160 else found

    @property
    def code_url(self) -> str | None:
        found = self.extra("code_url")
        if found:
            return found
        m = _GITHUB.search(f"{self.extra('comment') or ''} {self.abstract}")
        return m.group(0).rstrip(".") if m else None

    @property
    def source_names(self) -> list[str]:
        return list(dict.fromkeys(i.source for i in self.items))

    @property
    def discussions(self) -> list[tuple[str, str]]:
        return [(i.source, i.discussion_url) for i in self.items if i.discussion_url]

    @property
    def canonical_urls(self) -> set[str]:
        return {i.canonical_url for i in self.items if i.canonical_url}

    @property
    def paper_keys(self) -> set[str]:
        keys = {f"arxiv:{i.arxiv_id}" for i in self.items if i.arxiv_id}
        keys |= {f"doi:{i.doi}" for i in self.items if i.doi}
        return keys

    @property
    def newest(self) -> datetime | None:
        stamps = [i.timestamp for i in self.items if i.timestamp]
        return max(stamps) if stamps else None

    def metric(self, name: str) -> float:
        return max((i.metrics.get(name, 0.0) for i in self.items), default=0.0)

    def extra(self, key: str) -> Any:
        return next((i.extra[key] for i in self._papers_first if i.extra.get(key)), None)

    @property
    def _papers_first(self) -> list[Item]:
        return sorted(self.items, key=lambda i: (0 if i.kind == PAPER else 1, i.tier))


@dataclass
class DigestSection:
    key: str
    title: str
    clusters: list[Cluster]


@dataclass
class Digest:
    date: str  # YYYY-MM-DD in the configured timezone
    sections: list[DigestSection]
    candidates: int = 0  # how many clusters were considered

    @property
    def count(self) -> int:
        return sum(len(s.clusters) for s in self.sections)

    @property
    def subject(self) -> str:
        return f"Academic Digest — {self.date} ({self.count} papers)"
