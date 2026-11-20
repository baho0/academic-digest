"""Render a Digest to HTML + plain text (Jinja2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import Digest
from ..pipeline.normalize import domain

_TEMPLATES = Path(__file__).parent / "templates"


@dataclass
class Rendered:
    subject: str
    html: str
    text: str


def short_authors(authors: list[str], limit: int = 3) -> str:
    if not authors:
        return ""
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + " et al."


def make_env(*dirs: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader([*dirs, _TEMPLATES]),
        autoescape=select_autoescape(["html", "j2", "xml"], default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["domain"] = domain
    env.filters["short_authors"] = short_authors
    return env


def render(
    digest: Digest,
    category_titles: dict[str, str],
    explain: bool = False,
    web_base_url: str | None = None,
) -> Rendered:
    env = make_env()
    ctx = {
        "digest": digest,
        "categories": category_titles,
        "explain": explain,
        "web_url": f"{web_base_url.rstrip('/')}/digests/{digest.date}" if web_base_url else None,
    }
    return Rendered(
        subject=digest.subject,
        html=env.get_template("digest.html.j2").render(**ctx),
        text=env.get_template("digest.txt.j2").render(**ctx),
    )
