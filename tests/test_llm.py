"""LLM infrastructure with a fake provider: registry, cache, budget, prompts, hybrid scoring, summaries."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from academicdigest import db
from academicdigest.llm import (
    BudgetedLLM,
    CachedLLM,
    LLMBudgetExceeded,
    LLMError,
    LLMRequest,
    LLMResponse,
    build_llm,
    parse_json,
    register_provider,
    render_prompt,
)
from academicdigest.llm.cache import usage
from academicdigest.models import Cluster
from academicdigest.pipeline import run as pipeline
from academicdigest.pipeline.score import HybridScorer, RuleScorer, build_scorer
from academicdigest.pipeline.summarize import LLMSummarizer, build_summarizer

from .conftest import NOW, FakeNotifier, make_item, make_settings
from .test_digest_flow import seed


class ScriptedLLM:
    """Answers with `reply(request)`; counts calls."""

    name = "fake"

    def __init__(self, reply=None, model: str = "fake-1") -> None:
        self.model = model
        self.reply = reply or (lambda req: "ok")
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        text = self.reply(request)
        if isinstance(text, Exception):
            raise text
        return LLMResponse(text=text, model=self.model, input_tokens=10, output_tokens=3)


@register_provider("scripted-test")
def _factory(settings):
    return ScriptedLLM(lambda req: '{"score": 9, "reason": "core topic"}' if req.json else "A short summary.")


def paper(title="Speculative decoding at scale", **kw) -> Cluster:
    return Cluster(items=[make_item(title, source="arXiv", tier=3, arxiv_id=kw.pop("arxiv_id", "2611.00001"),
                                    excerpt="We make decoding 3x faster.", authors=["A. B."], **kw)])


def test_registry():
    assert build_llm(make_settings()) is None  # provider "none"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm(make_settings(llm_provider="nope"))
    client = build_llm(make_settings(llm_provider="scripted-test"))
    assert isinstance(client, BudgetedLLM) and client.name == "fake"


def test_cache_returns_stored_answer_and_logs_usage():
    inner = ScriptedLLM()
    with Session(db.make_engine("sqlite://")) as session:
        llm = CachedLLM(inner, session)
        req = LLMRequest(prompt="hi", purpose="summary")
        first, second = llm.complete(req), llm.complete(req)
        assert (first.cached, second.cached) == (False, True) and second.text == "ok"
        assert len(inner.calls) == 1
        llm.complete(LLMRequest(prompt="hi", temperature=0.5))  # different request -> new call
        assert len(inner.calls) == 2
        assert ("fake", "fake-1", "summary", 1, 10, 3) in usage(session)


def test_budget_counts_only_paid_calls():
    with Session(db.make_engine("sqlite://")) as session:
        llm = CachedLLM(BudgetedLLM(ScriptedLLM(), max_calls=1), session)
        llm.complete(LLMRequest(prompt="a"))
        llm.complete(LLMRequest(prompt="a"))  # cached: free
        with pytest.raises(LLMBudgetExceeded):
            llm.complete(LLMRequest(prompt="b"))


def test_prompts_render_system_and_user_parts():
    system, prompt = render_prompt("rate", paper=paper(), profile="Likes fast inference.")
    assert "JSON" in system and "Speculative decoding at scale" in prompt and "Likes fast inference." in prompt
    system, prompt = render_prompt("summarize", paper=paper(), max_chars=300, language="Turkish")
    assert "Turkish" in system and "We make decoding 3x faster." in prompt


@pytest.mark.parametrize("text", [
    '{"score": 7}', '```json\n{"score": 7}\n```', 'Sure! {"score": 7} Hope this helps.',
])
def test_parse_json_tolerates_wrapping(text):
    assert parse_json(text) == {"score": 7}


def test_parse_json_rejects_garbage():
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_hybrid_scorer_adjusts_top_candidates(scoring_cfg):
    ratings = {"Speculative decoding at scale": '{"score": 9, "reason": "core"}', "Boring": "not json"}
    llm = ScriptedLLM(lambda req: next(v for k, v in ratings.items() if k in req.prompt))
    good, bad = paper(), paper("Boring", arxiv_id="2611.00002")
    HybridScorer(RuleScorer(scoring_cfg), llm, scoring_cfg).score([good, bad], NOW)
    assert good.breakdown[-1].label == "llm 9/10" and good.breakdown[-1].delta == 4.0
    assert good.notes["llm_reason"] == "core"
    assert not any(b.label.startswith("llm") for b in bad.breakdown)  # failure keeps rule score
    assert all(r.json and r.purpose == "rating" for r in llm.calls)


def test_hybrid_scorer_stops_at_budget(scoring_cfg):
    llm = ScriptedLLM(lambda req: LLMBudgetExceeded("budget"))
    clusters = [paper(), paper("Other", arxiv_id="2611.00002")]
    HybridScorer(RuleScorer(scoring_cfg), llm, scoring_cfg).score(clusters, NOW)
    assert len(llm.calls) == 1


def test_llm_summarizer_falls_back_to_excerpt():
    llm = ScriptedLLM(lambda req: LLMError("down") if "Broken" in req.prompt else "Concise summary.")
    ok, broken = paper(), paper("Broken one", arxiv_id="2611.00002")
    LLMSummarizer(llm).summarize([ok, broken])
    assert ok.summary == "Concise summary."
    assert broken.summary == "We make decoding 3x faster."


def test_engines_need_a_client(scoring_cfg):
    scoring_cfg["engine"], scoring_cfg["summarizer"] = "hybrid", "llm"
    with pytest.raises(ValueError, match="AD_LLM_PROVIDER"):
        build_scorer(scoring_cfg)
    with pytest.raises(ValueError, match="AD_LLM_PROVIDER"):
        build_summarizer(scoring_cfg)


def test_digest_with_llm_end_to_end():
    s = make_settings(llm_provider="scripted-test")
    s.scoring.update(engine="hybrid", summarizer="llm")
    with Session(db.make_engine(s.database_url)) as session:
        seed(session)
        notifier = FakeNotifier()
        assert pipeline.run_digest(s, session, notifier, NOW, llm=build_llm(s, session)) == "sent"
        html = notifier.sent[0].html
        assert "A short summary." in html
        row = db.get_digest(session, pipeline.local_date(s, NOW))
        top = max(row.clusters, key=lambda c: c.score)
        assert ["llm 9/10", 4.0] in top.breakdown and top.meta["notes"]["llm_reason"] == "core topic"
        # A rebuild hits the cache: no new rows in llm_calls
        before = sum(r[3] for r in usage(session))
        pipeline.build_digest(s, session, NOW + timedelta(minutes=5), exclude_sent=False, llm=build_llm(s, session))
        assert sum(r[3] for r in usage(session)) == before
