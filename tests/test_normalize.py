import pytest

from academicdigest.pipeline.normalize import (
    arxiv_id_from,
    canonical_url,
    doi_from_url,
    domain,
    fold_name,
    normalize_doi,
    split_authors,
    strip_html,
    truncate,
)


@pytest.mark.parametrize("text,expected", [
    ("https://arxiv.org/abs/2611.24943", "2611.24943"),
    ("https://arxiv.org/abs/2611.24943v3", "2611.24943"),
    ("http://arxiv.org/pdf/2611.24943v2", "2611.24943"),
    ("https://arxiv.org/html/2611.24943v1#S3", "2611.24943"),
    ("arXiv:2611.24943v1", "2611.24943"),
    ("2611.24943", "2611.24943"),
    ("https://arxiv.org/abs/hep-th/9901001v1", "hep-th/9901001"),
    ("10.48550/arXiv.2611.24943", "2611.24943"),
    ("https://example.com/2611.24943", None),
    (None, None),
])
def test_arxiv_id_from(text, expected):
    assert arxiv_id_from(text) == expected


def test_canonical_url_collapses_arxiv_variants():
    urls = ["https://arxiv.org/pdf/2611.24943v2", "http://www.arxiv.org/abs/2611.24943?context=cs"]
    assert {canonical_url(u) for u in urls} == {"https://arxiv.org/abs/2611.24943"}


def test_canonical_url_strips_tracking_and_noise():
    a = canonical_url("http://www.Example.com/news/story/?utm_source=x&id=5&ref=hn#comments")
    b = canonical_url("https://example.com/news/story?id=5")
    assert a == b == "https://example.com/news/story?id=5"


def test_doi_helpers():
    assert normalize_doi("https://doi.org/10.1038/S41586-026-11445-Z") == "10.1038/s41586-026-11445-z"
    assert normalize_doi("doi:10.1145/3830454.3832631.") == "10.1145/3830454.3832631"
    assert normalize_doi("no doi here") is None
    assert doi_from_url("https://www.nature.com/articles/s41586-026-11445-z") == "10.1038/s41586-026-11445-z"
    assert doi_from_url("https://example.com/paper") is None


def test_text_helpers():
    assert domain("https://www.nature.com/x") == "nature.com"
    assert strip_html("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"
    text = "First sentence here. Second sentence is much longer and goes on and on."
    assert truncate(text, 36) == "First sentence here."
    assert fold_name("Véronique  Gouverneur") == fold_name("veronique gouverneur")
    assert split_authors("Ada Lovelace, Alan Turing and Grace Hopper") == ["Ada Lovelace", "Alan Turing", "Grace Hopper"]
