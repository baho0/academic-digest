"""Text / URL / identifier normalisation helpers."""

from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = re.compile(
    r"^(utm_.*|ref|ref_src|ref_url|source|src|fbclid|gclid|mc_cid|mc_eid|"
    r"cmpid|cmp|campaign|ocid|smid|sr_share|share|s|taid|guccounter|via|feed|"
    r"rss|mod|tpcc|ito|at_medium|at_campaign)$",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# New-style (2007+) and old-style (hep-th/9901001) arXiv identifiers.
_ARXIV_NEW = r"\d{4}\.\d{4,5}"
_ARXIV_OLD = r"[a-z][a-z\-]+(?:\.[A-Z]{2})?/\d{7}"
_ARXIV_URL = re.compile(
    rf"arxiv\.org/(?:abs|pdf|html|format)/({_ARXIV_NEW}|{_ARXIV_OLD})(?:v\d+)?", re.I
)
_ARXIV_BARE = re.compile(rf"^(?:arxiv:)?({_ARXIV_NEW}|{_ARXIV_OLD})(?:v\d+)?$", re.I)
_ARXIV_DOI = re.compile(rf"^10\.48550/arxiv\.({_ARXIV_NEW}|{_ARXIV_OLD})$", re.I)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.I)
_NATURE_ARTICLE = re.compile(r"nature\.com/articles/([a-z]\d{5}-\d{3}-[\w-]+)", re.I)


def canonical_url(url: str) -> str:
    """Normalise a URL so the same paper from different sources compares equal."""
    if not url:
        return ""
    aid = arxiv_id_from(url)
    if aid:
        return f"https://arxiv.org/abs/{aid}"
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m.") or host.startswith("amp."):
        host = host.split(".", 1)[1]
    path = re.sub(r"/(amp|index\.html?)/?$", "/", parts.path or "/")
    path = path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=False)
            if not _TRACKING_PARAMS.match(k)
        )
    )
    return urlunsplit(("https", host, path, query, ""))


def arxiv_id_from(text: str | None) -> str | None:
    """arXiv id (version stripped) from a URL, 'arXiv:XXXX' string, bare id or arXiv DOI."""
    if not text:
        return None
    text = text.strip()
    for pattern in (_ARXIV_URL, _ARXIV_BARE, _ARXIV_DOI):
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def normalize_doi(text: str | None) -> str | None:
    """'https://doi.org/10.1038/X', 'doi:10.1038/X' -> '10.1038/x' (DOIs are case-insensitive)."""
    if not text:
        return None
    m = _DOI.search(unquote(text))
    if not m:
        return None
    return m.group(1).rstrip(".,;)").lower()


def doi_from_url(url: str | None) -> str | None:
    """DOI embedded in a publisher URL (doi.org, nature.com/articles/...)."""
    if not url:
        return None
    m = _NATURE_ARTICLE.search(url)
    if m:
        return f"10.1038/{m.group(1).lower()}"
    if "doi" in url.lower():
        return normalize_doi(url)
    return None


def domain(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str, limit: int = 300) -> str:
    """Cut text to ~limit chars, preferring a sentence boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx >= limit * 0.5:
            return cut[: idx + 1]
    idx = cut.rfind(" ")
    return (cut[:idx] if idx > 0 else cut).rstrip(",;:—-") + "…"


def clean_title(title: str) -> str:
    return _WS_RE.sub(" ", html.unescape(title or "")).strip()


def fold_name(name: str) -> str:
    """'Véronique  Gouverneur' -> 'veronique gouverneur' (for author matching)."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", re.sub(r"[^\w\s-]", " ", name.lower())).strip()


def split_authors(value: str | None) -> list[str]:
    """'A, B and C' -> ['A', 'B', 'C']."""
    if not value:
        return []
    parts = re.split(r",|\band\b", value)
    return [p.strip() for p in parts if p.strip()]
