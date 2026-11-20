"""Web UI, JSON API and feed over a seeded in-memory database."""

from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from academicdigest import db
from academicdigest.pipeline import run as pipeline
from academicdigest.web import create_app

from .conftest import NOW, FakeNotifier, make_settings
from .test_digest_flow import seed

DATE = "2026-11-20"


def build(**settings_kw):
    s = make_settings(**settings_kw)
    engine = db.make_engine(s.database_url)
    with Session(engine) as session:
        seed(session)
        assert pipeline.run_digest(s, session, FakeNotifier(), NOW) == "sent"
    return TestClient(create_app(s, engine))


@pytest.fixture
def client():
    return build(web_base_url="https://papers.example.com")


def test_index_redirects_to_latest(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].endswith(f"/digests/{DATE}")


def test_empty_database_index():
    s = make_settings()
    c = TestClient(create_app(s, db.make_engine(s.database_url)))
    assert "No digests yet" in c.get("/").text


def test_digest_pages(client):
    page = client.get(f"/digests/{DATE}").text
    assert "Sparse experts make long-context models cheap" in page
    assert "incremental benchmark" not in page
    assert "Today&#39;s highlights" in page or "Today's highlights" in page

    everything = client.get(f"/digests/{DATE}?all=1").text
    assert "incremental benchmark" in everything and "hf_upvotes=45" in everything

    email = client.get(f"/digests/{DATE}/email").text
    assert email.startswith("<!doctype html>") and "https://papers.example.com/digests/" in email
    assert client.get("/digests/1999-01-01").status_code == 404


def test_papers_search_and_detail(client):
    page = client.get("/papers", params={"q": "sparse experts", "days": 0}).text
    assert page.count('class="paper"') == 1  # arXiv + HF rows are one paper
    assert "benchmark" not in page
    assert f"digest {DATE}" in page

    data = client.get("/api/papers", params={"q": "Lovelace", "days": 0}).json()
    assert [p["arxiv_id"] for p in data] == ["2611.00001"]
    pid = data[0]["id"]

    detail = client.get(f"/api/papers/{pid}").json()
    assert {s["source"] for s in detail["sources"]} == {"arXiv", "HF Papers"}
    assert detail["digests"][0]["date"] == DATE
    assert "Seen in" in client.get(f"/papers/{pid}").text
    assert client.get("/api/papers/999999").status_code == 404


def test_api_digests(client):
    rows = client.get("/api/digests").json()
    assert rows[0]["date"] == DATE and rows[0]["status"] == "sent" and rows[0]["selected"] == 1

    d = client.get(f"/api/digests/{DATE}").json()
    assert d["sections"][0]["key"] == "top"
    paper = d["sections"][0]["papers"][0]
    assert paper["arxiv_id"] == "2611.00001" and paper["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert {"label", "delta"} <= set(paper["breakdown"][0])
    assert "all_candidates" not in d

    full = client.get("/api/digests/latest", params={"all": True}).json()
    assert len(full["all_candidates"]) == 2


def test_status_and_feed(client):
    status = client.get("/api/status").json()
    assert {s["source"] for s in status["sources"]} == {"arXiv", "HF Papers"}
    assert "HF Papers" in client.get("/status").text

    r = client.get("/feed.xml")
    assert r.headers["content-type"].startswith("application/atom+xml")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = ElementTree.fromstring(r.content).findall("a:entry", ns)
    assert len(entries) == 1
    assert entries[0].find("a:link", ns).get("href") == f"https://papers.example.com/digests/{DATE}"
    assert "Sparse experts" in entries[0].find("a:content", ns).text


def test_openapi_and_static(client):
    assert "/api/papers" in client.get("/api/openapi.json").json()["paths"]
    assert client.get("/static/style.css").status_code == 200


def test_basic_auth():
    c = build(web_user="me", web_password="secret")
    assert c.get("/healthz").status_code == 200  # monitoring stays open
    assert c.get(f"/digests/{DATE}").status_code == 401
    assert c.get("/api/digests", auth=("me", "wrong")).status_code == 401
    assert c.get("/api/digests", auth=("me", "secret")).status_code == 200
