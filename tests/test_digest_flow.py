"""End-to-end: store items -> enrich -> build -> send -> idempotency -> no repeats."""

from datetime import timedelta

from sqlalchemy.orm import Session

from academicdigest import db
from academicdigest.models import MENTION, RawItem
from academicdigest.pipeline import run as pipeline

from .conftest import NOW, FakeNotifier, make_settings


def seed(session: Session) -> None:
    raws = [
        RawItem(source="arXiv", guid="2611.00001", title="Sparse experts make long-context models cheap",
                url="https://arxiv.org/abs/2611.00001", tier=3, arxiv_id="2611.00001", subjects=["cs.LG"],
                authors=["Ada Lovelace", "Alan Turing"], published_at=NOW - timedelta(hours=3),
                excerpt="We show that sparse experts cut inference cost by 4x. Second sentence."),
        RawItem(source="HF Papers", guid="2611.00001", title="Sparse experts make long-context models cheap",
                url="https://arxiv.org/abs/2611.00001", arxiv_id="2611.00001", category_hint="ml",
                discussion_url="https://huggingface.co/papers/2611.00001",
                published_at=NOW - timedelta(hours=2), metrics={"hf_upvotes": 45}),
        RawItem(source="arXiv", guid="2611.00002", title="Yet another incremental benchmark",
                url="https://arxiv.org/abs/2611.00002", tier=3, arxiv_id="2611.00002", subjects=["cs.CL"],
                published_at=NOW - timedelta(hours=3), excerpt="We add 0.1 points."),
    ]
    db.upsert_items(session, raws)


def test_full_flow():
    s = make_settings()
    engine = db.make_engine(s.database_url)
    with Session(engine) as session:
        seed(session)
        notifier = FakeNotifier()
        assert pipeline.run_digest(s, session, notifier, NOW) == "sent"
        assert len(notifier.sent) == 1
        msg = notifier.sent[0]
        assert "Sparse experts make long-context models cheap" in msg.html and "Sparse experts" in msg.text
        assert "incremental benchmark" not in msg.html
        assert "Ada Lovelace, Alan Turing" in msg.html
        assert "cut inference cost by 4x" in msg.html  # abstract excerpt
        assert "45 ▲" in msg.html

        row = db.get_digest(session, pipeline.local_date(s, NOW))
        assert row.status == "sent" and row.candidates == 2
        picked = [c for c in row.clusters if c.selected]
        assert len(picked) == 1 and picked[0].meta["arxiv_id"] == "2611.00001"
        assert {i.source for i in picked[0].items} == {"arXiv", "HF Papers"}

        # Same day again -> skipped
        assert pipeline.run_digest(s, session, notifier, NOW) == "skipped"
        assert len(notifier.sent) == 1

        # Next day an HN thread about the same paper appears: still not repeated
        db.upsert_items(session, [RawItem(
            source="Hacker News", guid="99", kind=MENTION, title="Sparse experts paper [pdf]",
            url="https://arxiv.org/pdf/2611.00001v2", arxiv_id="2611.00001",
            published_at=NOW + timedelta(hours=20), metrics={"hn_points": 400})])
        digest, _ = pipeline.build_digest(s, session, NOW + timedelta(hours=24))
        assert digest.count == 0


def test_failed_send_is_recorded_and_retried(monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    s = make_settings()
    with Session(db.make_engine(s.database_url)) as session:
        seed(session)
        try:
            pipeline.run_digest(s, session, FakeNotifier(fail=True), NOW)
        except ConnectionError:
            pass
        row = db.get_digest(session, pipeline.local_date(s, NOW))
        assert row.status == "failed" and "smtp down" in row.error
        assert pipeline.run_digest(s, session, FakeNotifier(), NOW) == "sent"


def test_empty_day_not_sent_when_configured():
    s = make_settings()
    s.scoring["selection"]["send_empty"] = False
    with Session(db.make_engine(s.database_url)) as session:
        notifier = FakeNotifier()
        assert pipeline.run_digest(s, session, notifier, NOW) == "empty"
        assert notifier.sent == []
        assert db.get_digest(session, pipeline.local_date(s, NOW)).status == "empty"


def test_enrichment_survives_refetch():
    with Session(db.make_engine("sqlite://")) as session:
        raw = RawItem(source="arXiv", guid="1", title="t", url="https://arxiv.org/abs/2611.00001",
                      arxiv_id="2611.00001", published_at=NOW, metrics={"x": 1})
        assert db.upsert_items(session, [raw]) == (1, 0)
        item = db.items_since(session, NOW - timedelta(hours=1))[0]
        db.patch_items(session, {item.id: db.ItemPatch(extra={"comment": "Accepted at ICML"},
                                                       authors=["Filled In"])})
        raw.metrics = {"x": 5}
        assert db.upsert_items(session, [raw]) == (0, 1)
        item = db.items_since(session, NOW - timedelta(hours=1))[0]
        assert item.metrics["x"] == 5
        assert item.extra["comment"] == "Accepted at ICML"
        assert item.authors == ["Filled In"]
