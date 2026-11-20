import pytest

from academicdigest.models import Cluster
from academicdigest.pipeline.classify import KeywordClassifier
from academicdigest.pipeline.score import RuleScorer

from .conftest import NOW, make_item


def scored(cfg, *items, source_weights=None):
    c = Cluster(items=list(items))
    c.category = KeywordClassifier(cfg["categories"]).classify(c)
    RuleScorer(cfg, source_weights).score([c], NOW)
    return c


def labels(c):
    return {b.label for b in c.breakdown}


def arxiv(title="A study of something", **kw):
    kw.setdefault("subjects", ["cs.LG"])
    return make_item(title, source="arXiv", tier=3, arxiv_id=kw.pop("arxiv_id", "2611.00001"), **kw)


def test_plain_arxiv_paper_does_not_qualify(scoring_cfg):
    c = scored(scoring_cfg, arxiv(excerpt="We propose a method and evaluate it."))
    assert c.score < scoring_cfg["selection"]["min_score"]


def test_community_attention_qualifies(scoring_cfg):
    c = scored(
        scoring_cfg,
        arxiv(),
        make_item("A study of something", source="HF Papers", arxiv_id="2611.00001", metrics={"hf_upvotes": 20}),
    )
    assert "coverage x2" not in labels(c)  # arXiv is a firehose, not an endorsement
    assert c.score >= scoring_cfg["selection"]["min_score"]


def test_independent_sources_add_coverage(scoring_cfg):
    c = scored(
        scoring_cfg,
        make_item("X", source="HF Papers", arxiv_id="2611.00001", metrics={"hf_upvotes": 20}),
        make_item("X", source="Hacker News", arxiv_id="2611.00001", metrics={"hn_points": 200}),
    )
    assert "coverage x2" in labels(c)


def test_top_venue_and_code_from_comment(scoring_cfg):
    c = scored(scoring_cfg, arxiv(extra={"comment": "Accepted at ICLR 2027 (oral). Code: https://github.com/a/b"}))
    assert {"top-venue", "highlight", "accepted", "code"} <= labels(c)
    assert c.code_url == "https://github.com/a/b"
    assert c.venue.startswith("Accepted at ICLR")


def test_page_count_comment_is_not_a_venue(scoring_cfg):
    c = scored(scoring_cfg, arxiv(extra={"comment": "12 pages, 5 figures"}))
    assert c.venue is None and "top-venue" not in labels(c)


@pytest.mark.parametrize("title,extra,label", [
    ("Some paper", {"comment": "This paper has been withdrawn"}, "withdrawn"),
    ("Author Correction: Some earlier result", {}, "correction"),
    ("Retraction Note: Detection of anisotropic structures", {}, "correction"),
    ("Daily briefing: what we learned", {}, "not-research"),
])
def test_penalties(scoring_cfg, title, extra, label):
    c = scored(scoring_cfg, make_item(title, source="Nature", tier=1, category_hint="science", extra=extra))
    assert label in labels(c)
    assert c.score < scoring_cfg["selection"]["min_score"]


def test_interest_title_full_abstract_half(scoring_cfg):
    in_title = scored(scoring_cfg, arxiv("Speculative decoding for faster serving"))
    in_abstract = scored(scoring_cfg, arxiv("A new method", excerpt="We speed up speculative decoding."))
    w = next(b.delta for b in in_title.breakdown if b.label == "interest:efficient-inference")
    w2 = next(b.delta for b in in_abstract.breakdown if b.label == "interest:efficient-inference")
    assert w == 2 * w2


def test_watched_author_ignores_accents_and_case(scoring_cfg):
    scoring_cfg["watch_authors"]["names"] = ["Veronique Gouverneur"]
    c = scored(scoring_cfg, arxiv(authors=["A. Person", "Véronique  GOUVERNEUR"]))
    assert "author:Véronique  GOUVERNEUR" in labels(c)


def test_classification_by_primary_subject(scoring_cfg):
    clf = KeywordClassifier(scoring_cfg["categories"])
    c = Cluster(items=[arxiv("Attacking image classifiers", subjects=["cs.CR", "cs.CV"])])
    assert clf.classify(c) == "security"
    c = Cluster(items=[arxiv("Attacking image classifiers", subjects=["cs.CR", "cs.CV"],
                             extra={"primary_subject": "cs.CV"})])
    assert clf.classify(c) == "vision"
    c = Cluster(items=[arxiv("Lasers", subjects=["physics.optics"])])
    assert clf.classify(c) == "science"


def test_acronym_keywords_are_case_sensitive(scoring_cfg):
    clf = KeywordClassifier(scoring_cfg["categories"])
    c = Cluster(items=[make_item("Girls in school sports")])
    clf.classify(c)
    assert c.topic_hits == 0  # "girls" must not match "RL"


def test_off_topic_mention(scoring_cfg):
    c = scored(scoring_cfg, make_item("The history of medieval bridges", source="Hacker News",
                                      metrics={"hn_points": 300}))
    assert "off-topic" in labels(c)


def test_freshness_decay(scoring_cfg):
    fresh = scored(scoring_cfg, make_item("X", source="Nature", tier=1, category_hint="science", hours_ago=1))
    stale = scored(scoring_cfg, make_item("X", source="Nature", tier=1, category_hint="science", hours_ago=120))
    assert stale.score < fresh.score


def test_source_weight_override(scoring_cfg):
    c = scored(scoring_cfg, make_item("X", source="Nature", tier=1), source_weights={"Nature": 6.0})
    assert c.breakdown[0].label == "Nature" and c.breakdown[0].delta == 6.0
