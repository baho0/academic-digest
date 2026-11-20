"""Storage layer (SQLAlchemy). SQLite today; any SQLAlchemy URL works.

Tables (the web UI in `academicdigest.web` reads the same data):
  items          every paper / mention ever collected (metrics refreshed on re-fetch,
                 metadata filled in by enrichers)
  digests        one row per day: status + rendered HTML/text
  clusters       every candidate paper considered for a digest, with score
                 breakdown; selected ones have section/rank set
  cluster_items  which items make up each cluster
  llm_calls      cache + usage log of every LLM completion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from .models import Cluster, Digest, Item, RawItem
from .pipeline.normalize import canonical_url


def utcnow() -> datetime:
    return datetime.now(UTC)


def _naive(dt: datetime | None) -> datetime | None:
    """Store UTC as naive datetimes (portable across SQLite/Postgres)."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=UTC) if dt is not None and dt.tzinfo is None else dt


class Base(DeclarativeBase):
    pass


class ItemRow(Base):
    __tablename__ = "items"
    __table_args__ = (UniqueConstraint("source", "guid", name="uq_items_source_guid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(100), index=True)
    guid: Mapped[str] = mapped_column(String(500))
    kind: Mapped[str] = mapped_column(String(20), default="paper")
    url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(String(1000), index=True)
    title: Mapped[str] = mapped_column(Text)
    excerpt: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    subjects: Mapped[list[str]] = mapped_column(JSON, default=list)
    arxiv_id: Mapped[str | None] = mapped_column(String(40), index=True)
    doi: Mapped[str | None] = mapped_column(String(300), index=True)
    tier: Mapped[int] = mapped_column(Integer, default=2)
    category_hint: Mapped[str | None] = mapped_column(String(50))
    discussion_url: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    def to_domain(self) -> Item:
        return Item(
            id=self.id,
            source=self.source,
            guid=self.guid,
            kind=self.kind,
            title=self.title,
            url=self.url,
            canonical_url=self.canonical_url,
            tier=self.tier,
            category_hint=self.category_hint,
            excerpt=self.excerpt,
            authors=list(self.authors or []),
            subjects=list(self.subjects or []),
            arxiv_id=self.arxiv_id,
            doi=self.doi,
            published_at=_aware(self.published_at),
            fetched_at=_aware(self.fetched_at),
            discussion_url=self.discussion_url,
            metrics=dict(self.metrics or {}),
            extra=dict(self.extra or {}),
        )


class DigestRow(Base):
    __tablename__ = "digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[str] = mapped_column(String(10), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|sent|failed|empty
    subject: Mapped[str] = mapped_column(Text, default="")
    html: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text)
    candidates: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime)

    clusters: Mapped[list[ClusterRow]] = relationship(
        back_populates="digest", cascade="all, delete-orphan", order_by="ClusterRow.id"
    )


class ClusterRow(Base):
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    digest_id: Mapped[int] = mapped_column(ForeignKey("digests.id"), index=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50))
    score: Mapped[float] = mapped_column(Float)
    breakdown: Mapped[list[Any]] = mapped_column(JSON, default=list)
    summary: Mapped[str | None] = mapped_column(Text)
    # Display fields (authors, venue, links...) so readers need no join.
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    section: Mapped[str | None] = mapped_column(String(50))
    rank: Mapped[int | None] = mapped_column(Integer)

    digest: Mapped[DigestRow] = relationship(back_populates="clusters")
    items: Mapped[list[ItemRow]] = relationship(secondary="cluster_items")


class ClusterItemRow(Base):
    __tablename__ = "cluster_items"

    cluster_id: Mapped[int] = mapped_column(ForeignKey("clusters.id"), primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), primary_key=True)


class LlmCallRow(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)  # sha256 of model + request
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str] = mapped_column(String(50), default="")
    response: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


# --------------------------------------------------------------------------- engine


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        path = url.removeprefix("sqlite:///")
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        # The web server reads from worker threads.
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, future=True, **kwargs)
    if url.startswith("sqlite:///") and not url.endswith(":memory:"):
        # Web server reads while collect/digest write: WAL lets them run side by side.
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=15000")
            cur.close()

    Base.metadata.create_all(engine)
    return engine


# --------------------------------------------------------------------------- items


def upsert_items(session: Session, raw_items: Iterable[RawItem]) -> tuple[int, int]:
    """Insert new items, refresh metrics/metadata of existing ones. Returns (new, updated)."""
    now = _naive(utcnow())
    new = updated = 0
    seen: set[tuple[str, str]] = set()
    for raw in raw_items:
        key = (raw.source, raw.guid)
        if key in seen:
            continue
        seen.add(key)
        row = session.scalar(
            select(ItemRow).where(ItemRow.source == raw.source, ItemRow.guid == raw.guid)
        )
        if row is None:
            session.add(
                ItemRow(
                    source=raw.source,
                    guid=raw.guid,
                    kind=raw.kind,
                    url=raw.url,
                    canonical_url=canonical_url(raw.url),
                    title=raw.title,
                    excerpt=raw.excerpt,
                    authors=raw.authors,
                    subjects=raw.subjects,
                    arxiv_id=raw.arxiv_id,
                    doi=raw.doi,
                    tier=raw.tier,
                    category_hint=raw.category_hint,
                    discussion_url=raw.discussion_url,
                    metrics=raw.metrics,
                    extra=raw.extra,
                    published_at=_naive(raw.published_at),
                    fetched_at=now,
                    updated_at=now,
                )
            )
            new += 1
        else:
            row.metrics = {**(row.metrics or {}), **raw.metrics}
            # Keep what enrichers added; the source's own fields win.
            row.extra = {**(row.extra or {}), **raw.extra}
            row.title = raw.title or row.title
            row.excerpt = raw.excerpt or row.excerpt
            row.authors = raw.authors or row.authors
            row.subjects = raw.subjects or row.subjects
            row.arxiv_id = raw.arxiv_id or row.arxiv_id
            row.doi = raw.doi or row.doi
            row.tier = raw.tier
            row.category_hint = raw.category_hint
            row.updated_at = now
            updated += 1
    session.commit()
    return new, updated


@dataclass
class ItemPatch:
    """What an enricher learned about an item. Empty fields are only filled, never overwritten."""

    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    excerpt: str | None = None
    authors: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    doi: str | None = None


def patch_items(session: Session, patches: dict[int, ItemPatch]) -> int:
    for item_id, p in patches.items():
        row = session.get(ItemRow, item_id)
        if row is None:
            continue
        row.metrics = {**(row.metrics or {}), **p.metrics}
        row.extra = {**(row.extra or {}), **p.extra}
        row.excerpt = row.excerpt or p.excerpt
        row.authors = row.authors or p.authors
        row.subjects = row.subjects or p.subjects
        row.doi = row.doi or p.doi
        row.updated_at = _naive(utcnow())
    session.commit()
    return len(patches)


def items_since(session: Session, since: datetime) -> list[Item]:
    """Items published (or, if undated, first fetched) after `since`."""
    since_n = _naive(since)
    rows = session.scalars(
        select(ItemRow).where(
            (ItemRow.published_at >= since_n)
            | ((ItemRow.published_at.is_(None)) & (ItemRow.fetched_at >= since_n))
        )
    ).all()
    return [r.to_domain() for r in rows]


# --------------------------------------------------------------------------- digests


def sent_paper_keys(session: Session, days: int = 30) -> set[str]:
    """Canonical URLs, arXiv ids and DOIs of every paper already delivered recently."""
    since = _naive(utcnow() - timedelta(days=days))
    rows = session.execute(
        select(ItemRow.canonical_url, ItemRow.arxiv_id, ItemRow.doi)
        .join(ClusterItemRow, ClusterItemRow.item_id == ItemRow.id)
        .join(ClusterRow, ClusterRow.id == ClusterItemRow.cluster_id)
        .join(DigestRow, DigestRow.id == ClusterRow.digest_id)
        .where(ClusterRow.selected.is_(True), DigestRow.status == "sent", DigestRow.created_at >= since)
    ).all()
    keys: set[str] = set()
    for url, aid, doi in rows:
        if url:
            keys.add(f"url:{url}")
        if aid:
            keys.add(f"arxiv:{aid}")
        if doi:
            keys.add(f"doi:{doi}")
    return keys


def sent_titles(session: Session, days: int = 30) -> list[str]:
    since = _naive(utcnow() - timedelta(days=days))
    rows = session.execute(
        select(ClusterRow.title)
        .join(DigestRow, DigestRow.id == ClusterRow.digest_id)
        .where(ClusterRow.selected.is_(True), DigestRow.status == "sent", DigestRow.created_at >= since)
    ).all()
    return [r[0] for r in rows]


def get_digest(session: Session, date: str) -> DigestRow | None:
    return session.scalar(select(DigestRow).where(DigestRow.date == date))


def cluster_meta(c: Cluster) -> dict[str, Any]:
    return {
        "authors": c.authors,
        "subjects": c.subjects,
        "venue": c.venue,
        "arxiv_id": c.arxiv_id,
        "doi": c.doi,
        "pdf_url": c.pdf_url,
        "code_url": c.code_url,
        "abstract": c.abstract,
        "sources": c.source_names,
        "discussions": c.discussions,
        "notes": c.notes,
        "metrics": {
            k: c.metric(k)
            for k in sorted({m for i in c.items for m in i.metrics})
        },
    }


def save_digest(
    session: Session,
    digest: Digest,
    candidates: list[Cluster],
    subject: str,
    html: str,
    text: str,
) -> DigestRow:
    """Create or replace the digest row for digest.date (status=pending)."""
    row = get_digest(session, digest.date)
    if row is None:
        row = DigestRow(date=digest.date, created_at=_naive(utcnow()))
        session.add(row)
    row.status = "pending"
    row.subject, row.html, row.text, row.error = subject, html, text, None
    row.candidates = digest.candidates
    row.created_at = _naive(utcnow())
    row.clusters.clear()
    session.flush()

    placement: dict[int, tuple[str, int]] = {}
    for section in digest.sections:
        for rank, c in enumerate(section.clusters, 1):
            placement[id(c)] = (section.key, rank)

    for c in candidates:
        sec, rank = placement.get(id(c), (None, None))
        cr = ClusterRow(
            title=c.title,
            url=c.url,
            category=c.category,
            score=round(c.score, 3),
            breakdown=[[b.label, round(b.delta, 3)] for b in c.breakdown],
            summary=c.summary,
            meta=cluster_meta(c),
            selected=sec is not None,
            section=sec,
            rank=rank,
        )
        cr.items = [session.get(ItemRow, i.id) for i in c.items if i.id]
        row.clusters.append(cr)
    session.commit()
    return row


def mark_digest(session: Session, row: DigestRow, status: str, error: str | None = None) -> None:
    row.status = status
    row.error = error
    if status == "sent":
        row.sent_at = _naive(utcnow())
    session.commit()
