"""Sources and enrichers against recorded / hand-made API responses (no network)."""

import json

import httpx

from academicdigest.enrich import build_enricher
from academicdigest.enrich.arxiv_meta import parse_api
from academicdigest.models import MENTION
from academicdigest.settings import EnricherConfig, SourceConfig
from academicdigest.sources import build_source

from .conftest import FIXTURES, make_item, make_settings, mock_client


def fetch(cfg: SourceConfig, handler):
    with mock_client(handler) as client:
        return build_source(cfg, client).fetch()


def test_arxiv_feed_keeps_new_and_cross_only():
    seen_urls = []

    def handler(request):
        seen_urls.append(str(request.url))
        return httpx.Response(200, content=(FIXTURES / "arxiv.xml").read_bytes())

    cfg = SourceConfig(name="arXiv", type="arxiv", tier=3, options={"categories": ["cs.CL", "cs.CR"]})
    items = fetch(cfg, handler)
    assert seen_urls == ["https://rss.arxiv.org/atom/cs.CL+cs.CR"]
    assert len(items) == 7  # 5 new + 2 cross; replace / replace-cross dropped
    assert {i.extra["announce_type"] for i in items} == {"new", "cross"}

    first = items[0]
    assert first.arxiv_id == "2611.24943" and first.guid == "2611.24943"
    assert first.url == "https://arxiv.org/abs/2611.24943"
    assert first.excerpt.startswith("Computational complexity theory")  # "arXiv:… Announce Type" stripped
    assert first.authors[:2] == ["Yu-Hung Wu", "Shao-Xiang Yuan"] and len(first.authors) == 6
    assert first.subjects[0] == "cs.CL" and "cs.LG" in first.subjects
    assert first.tier == 3

    with_doi = next(i for i in items if i.doi)
    assert with_doi.doi == "10.1007/s13042-025-02572-6"
    assert "Machine Learning and Cybernetics" in with_doi.extra["journal_ref"]


def test_journal_feed_filters_research_articles():
    cfg = SourceConfig(name="Nature", type="journal", tier=1, url="https://www.nature.com/nature.rss",
                       options={"url_include": "/articles/s41586"})
    items = fetch(cfg, lambda r: httpx.Response(200, content=(FIXTURES / "nature.xml").read_bytes()))
    assert len(items) == 3
    assert all("/articles/s41586" in i.url for i in items)
    retraction = next(i for i in items if i.title.startswith("Retraction"))
    assert retraction.doi == "10.1038/s41586-026-11388-5"
    calcium = next(i for i in items if "Calcium" in i.title)
    assert calcium.authors[0] == "Job J. C. Struijs" and len(calcium.authors) == 4
    assert calcium.extra["journal"] == "Nature"
    assert not (calcium.excerpt or "").startswith("Nature, Published online")


def test_hf_papers_threshold_and_metrics():
    def paper(pid, upvotes, repo=None):
        return {
            "paper": {"id": pid, "upvotes": upvotes, "title": f"Paper {pid}", "githubRepo": repo,
                      "authors": [{"name": "A. Author"}], "submittedOnDailyAt": "2026-11-20T00:00:00.000Z"},
            "title": f"Paper {pid}", "summary": "We do things.", "numComments": 2,
        }

    def handler(request):
        assert request.url.params["date"]
        return httpx.Response(200, json=[paper("2611.00001", 40, "https://github.com/a/b"), paper("2611.00002", 2)])

    cfg = SourceConfig(name="HF Papers", type="hf_papers", options={"min_upvotes": 10, "days": 2})
    items = fetch(cfg, handler)
    assert [i.arxiv_id for i in items] == ["2611.00001"]  # deduped across days, low votes dropped
    it = items[0]
    assert it.metrics == {"hf_upvotes": 40.0, "hf_comments": 2.0}
    assert it.extra["code_url"] == "https://github.com/a/b"
    assert it.discussion_url == "https://huggingface.co/papers/2611.00001"


def test_hackernews_mentions_link_to_papers():
    hits = [
        {"objectID": "1", "title": "Cool paper", "url": "https://arxiv.org/pdf/2611.00001v2",
         "created_at_i": 1795000000, "points": 120, "num_comments": 40},
        {"objectID": "2", "title": "Nature thing", "url": "https://www.nature.com/articles/s41586-026-00001-x",
         "created_at_i": 1795000000, "points": 50, "num_comments": 5},
    ]

    def handler(request):
        dom = request.url.params["query"]
        assert request.url.params["restrictSearchableAttributes"] == "url"
        return httpx.Response(200, json={"hits": [h for h in hits if dom in h["url"]]})

    cfg = SourceConfig(name="Hacker News", type="hackernews", options={"domains": ["arxiv.org", "nature.com"]})
    items = {i.guid: i for i in fetch(cfg, handler)}
    assert items["1"].kind == MENTION and items["1"].arxiv_id == "2611.00001"
    assert items["2"].doi == "10.1038/s41586-026-00001-x"
    assert items["1"].metrics["hn_points"] == 120


def test_arxiv_api_parse():
    meta = parse_api((FIXTURES / "arxiv_api.xml").read_bytes())
    assert set(meta) == {"2611.24943", "2611.24959"}
    assert meta["2611.24943"]["comment"] == "10 pages, 20 with Appendix"
    assert meta["2611.24943"]["subjects"][0] == "cs.CL"
    assert meta["2611.24959"]["doi"] == "10.1007/s13042-025-02572-6"


def test_arxiv_meta_enricher_patches_every_item_of_a_paper():
    requests = []

    def handler(request):
        requests.append(request.url.params["id_list"])
        return httpx.Response(200, content=(FIXTURES / "arxiv_api.xml").read_bytes())

    arxiv_item = make_item("LLM Complexity", id=1, source="arXiv", arxiv_id="2611.24943")
    hn_item = make_item("LLM Complexity (HN)", id=2, source="Hacker News", arxiv_id="2611.24943")
    done = make_item("Done", id=3, arxiv_id="2611.24959", extra={"arxiv_meta": True})
    cfg = EnricherConfig(name="arXiv metadata", type="arxiv_meta", options={"delay_seconds": 0})
    with mock_client(handler) as client:
        enricher = build_enricher(cfg, client, make_settings())
        todo = [i for i in (arxiv_item, hn_item, done) if enricher.wants(i)]
        patches = enricher.enrich(todo)
    assert requests == ["2611.24943"]
    assert set(patches) == {1, 2}
    assert patches[2].extra["comment"] == "10 pages, 20 with Appendix"
    assert patches[2].authors[0] == "Yu-Hung Wu"  # fills a bare HN mention


def test_semantic_scholar_enricher():
    def handler(request):
        ids = json.loads(request.content)["ids"]
        assert ids == ["ARXIV:2611.00001", "DOI:10.1/x"]
        return httpx.Response(200, json=[
            {"citationCount": 12, "influentialCitationCount": 1, "authors": [{"hIndex": 30}, {"hIndex": None}]},
            None,  # not indexed yet
        ])

    a = make_item("A", id=1, arxiv_id="2611.00001")
    b = make_item("B", id=2, doi="10.1/x")
    cfg = EnricherConfig(name="S2", type="semantic_scholar", options={})
    with mock_client(handler) as client:
        enricher = build_enricher(cfg, client, make_settings())
        assert enricher.wants(a) and not enricher.wants(make_item("no ids"))
        patches = enricher.enrich([a, b])
    assert patches[1].metrics == {"s2_citations": 12.0, "s2_influential": 1.0, "s2_max_hindex": 30.0}
    assert patches[2].metrics == {} and "s2_checked_at" in patches[2].extra
    b.extra.update(patches[2].extra)
    assert not enricher.wants(b)  # retried only after refresh_hours
