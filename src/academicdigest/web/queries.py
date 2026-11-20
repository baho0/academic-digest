"""Read-only queries behind the web UI and JSON API.

Everything returns plain dicts so the HTML templates and the API share one shape.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import String, case, cast, func, or_, select
from sqlalchemy.orm import Session

from ..db import ClusterItemRow, ClusterRow, DigestRow, ItemRow, _aware, _naive, utcnow


def _iso(dt) -> str | None:
    dt = _aware(dt)
    return dt.isoformat() if dt else None


def paper_key(row: ItemRow) -> str:
    if row.arxiv_id:
        return f"arxiv:{row.arxiv_id}"
    if row.doi:
        return f"doi:{row.doi}"
    return f"url:{row.canonical_url}"


# --------------------------------------------------------------------------- digests


def digest_summary(row: DigestRow, selected: int | None = None) -> dict[str, Any]:
    return {
        "date": row.date,
        "status": row.status,
        "subject": row.subject,
        "candidates": row.candidates,
        "selected": selected if selected is not None else sum(1 for c in row.clusters if c.selected),
        "created_at": _iso(row.created_at),
        "sent_at": _iso(row.sent_at),
        "error": row.error,
    }


def list_digests(session: Session, limit: int = 60, offset: int = 0, sent_only: bool = False) -> list[dict]:
    selected = (
        select(ClusterRow.digest_id, func.count().label("n"))
        .where(ClusterRow.selected.is_(True))
        .group_by(ClusterRow.digest_id)
        .subquery()
    )
    q = (
        select(DigestRow, func.coalesce(selected.c.n, 0))
        .outerjoin(selected, selected.c.digest_id == DigestRow.id)
        .order_by(DigestRow.date.desc())
        .limit(limit)
        .offset(offset)
    )
    if sent_only:
        q = q.where(DigestRow.status == "sent")
    return [digest_summary(row, n) for row, n in session.execute(q).all()]


def latest_digest_date(session: Session) -> str | None:
    return session.scalar(select(DigestRow.date).order_by(DigestRow.date.desc()).limit(1))


def cluster_dict(c: ClusterRow, category_titles: dict[str, str]) -> dict[str, Any]:
    meta = c.meta or {}
    return {
        "id": c.id,
        "title": c.title,
        "url": c.url,
        "category": c.category,
        "category_title": category_titles.get(c.category, c.category),
        "score": c.score,
        "breakdown": [{"label": label, "delta": delta} for label, delta in c.breakdown or []],
        "summary": c.summary,
        "selected": c.selected,
        "section": c.section,
        "rank": c.rank,
        "item_ids": [i.id for i in c.items],
        **{k: meta.get(k) for k in (
            "authors", "subjects", "venue", "arxiv_id", "doi", "pdf_url", "code_url",
            "abstract", "sources", "discussions", "metrics", "notes",
        )},
    }


def digest_detail(
    session: Session, date: str, category_titles: dict[str, str], top_title: str, include_all: bool = False
) -> dict[str, Any] | None:
    row = session.scalar(select(DigestRow).where(DigestRow.date == date))
    if row is None:
        return None
    papers = [cluster_dict(c, category_titles) for c in row.clusters]
    order = ["top", *category_titles]
    sections = []
    for key in order:
        picked = sorted((p for p in papers if p["section"] == key), key=lambda p: p["rank"] or 0)
        if picked:
            sections.append({
                "key": key,
                "title": top_title if key == "top" else category_titles.get(key, key),
                "papers": picked,
            })
    out = {**digest_summary(row), "sections": sections}
    if include_all:
        out["all_candidates"] = sorted(papers, key=lambda p: p["score"], reverse=True)
    dates = session.execute(
        select(DigestRow.date).where(DigestRow.date != date).order_by(DigestRow.date)
    ).scalars().all()
    out["prev"] = max((d for d in dates if d < date), default=None)
    out["next"] = min((d for d in dates if d > date), default=None)
    return out


def digest_email_html(session: Session, date: str) -> str | None:
    return session.scalar(select(DigestRow.html).where(DigestRow.date == date))


def feed_digests(session: Session, limit: int = 20) -> list[DigestRow]:
    return list(
        session.scalars(
            select(DigestRow).where(DigestRow.status == "sent").order_by(DigestRow.date.desc()).limit(limit)
        )
    )


# --------------------------------------------------------------------------- papers


def item_dict(row: ItemRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "source": row.source,
        "kind": row.kind,
        "title": row.title,
        "url": row.url,
        "abstract": row.excerpt,
        "authors": row.authors or [],
        "subjects": row.subjects or [],
        "arxiv_id": row.arxiv_id,
        "doi": row.doi,
        "discussion_url": row.discussion_url,
        "metrics": row.metrics or {},
        "comment": (row.extra or {}).get("comment"),
        "journal_ref": (row.extra or {}).get("journal_ref") or (row.extra or {}).get("journal"),
        "code_url": (row.extra or {}).get("code_url"),
        "published_at": _iso(row.published_at),
        "fetched_at": _iso(row.fetched_at),
    }


def _appearances(session: Session, item_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """item id -> digests where a cluster containing it was selected."""
    if not item_ids:
        return {}
    rows = session.execute(
        select(ClusterItemRow.item_id, DigestRow.date, ClusterRow.section, ClusterRow.score)
        .join(ClusterRow, ClusterRow.id == ClusterItemRow.cluster_id)
        .join(DigestRow, DigestRow.id == ClusterRow.digest_id)
        .where(ClusterItemRow.item_id.in_(item_ids), ClusterRow.selected.is_(True))
        .order_by(DigestRow.date.desc())
    ).all()
    out: dict[int, list[dict[str, Any]]] = {}
    for item_id, date, section, score in rows:
        out.setdefault(item_id, []).append({"date": date, "section": section, "score": score})
    return out


def search_papers(
    session: Session,
    q: str = "",
    days: int | None = 30,
    source: str | None = None,
    in_digest: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Papers (not mentions), newest first, one row per paper even if several sources had it."""
    stmt = select(ItemRow).where(ItemRow.kind == "paper")
    if q.strip():
        for term in q.split():
            like = f"%{term}%"
            stmt = stmt.where(
                or_(
                    ItemRow.title.ilike(like),
                    ItemRow.excerpt.ilike(like),
                    cast(ItemRow.authors, String).ilike(like),
                    ItemRow.arxiv_id == term,
                )
            )
    if days:
        stmt = stmt.where(ItemRow.fetched_at >= _naive(utcnow() - timedelta(days=days)))
    if source:
        stmt = stmt.where(ItemRow.source == source)
    if in_digest:
        stmt = stmt.where(
            ItemRow.id.in_(
                select(ClusterItemRow.item_id)
                .join(ClusterRow, ClusterRow.id == ClusterItemRow.cluster_id)
                .where(ClusterRow.selected.is_(True))
            )
        )
    stmt = stmt.order_by(func.coalesce(ItemRow.published_at, ItemRow.fetched_at).desc(), ItemRow.id.desc())

    # De-duplicate by paper identity; over-fetch so a page stays full.
    seen: set[str] = set()
    rows: list[ItemRow] = []
    skipped = 0
    for row in session.scalars(stmt.limit((offset + limit) * 3)):
        key = paper_key(row)
        if key in seen:
            continue
        seen.add(key)
        if skipped < offset:
            skipped += 1
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    apps = _appearances(session, [r.id for r in rows])
    return [{**item_dict(r), "digests": apps.get(r.id, [])} for r in rows]


def paper_detail(session: Session, item_id: int) -> dict[str, Any] | None:
    row = session.get(ItemRow, item_id)
    if row is None:
        return None
    same = [row]
    conds = []
    if row.arxiv_id:
        conds.append(ItemRow.arxiv_id == row.arxiv_id)
    if row.doi:
        conds.append(ItemRow.doi == row.doi)
    if conds:
        same = list(session.scalars(select(ItemRow).where(or_(*conds)).order_by(ItemRow.id)))
    apps = _appearances(session, [r.id for r in same])
    seen_dates: set[str] = set()
    digests = []
    for r in same:
        for a in apps.get(r.id, []):
            if a["date"] not in seen_dates:
                seen_dates.add(a["date"])
                digests.append(a)
    lead = next((r for r in same if r.kind == "paper" and r.excerpt), row)
    return {**item_dict(lead), "sources": [item_dict(r) for r in same], "digests": digests}


# --------------------------------------------------------------------------- status


def source_stats(session: Session) -> list[dict[str, Any]]:
    now = utcnow()
    day, week = _naive(now - timedelta(days=1)), _naive(now - timedelta(days=7))
    rows = session.execute(
        select(
            ItemRow.source,
            func.count(),
            func.sum(case((ItemRow.fetched_at >= day, 1), else_=0)),
            func.sum(case((ItemRow.fetched_at >= week, 1), else_=0)),
            func.max(ItemRow.fetched_at),
            func.max(ItemRow.updated_at),
        ).group_by(ItemRow.source).order_by(ItemRow.source)
    ).all()
    return [
        {
            "source": src,
            "total": total,
            "new_24h": int(d or 0),
            "new_7d": int(w or 0),
            "last_new_at": _iso(last_new),
            "last_seen_at": _iso(last_seen),
        }
        for src, total, d, w, last_new, last_seen in rows
    ]

