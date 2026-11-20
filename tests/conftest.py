from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml

from academicdigest.models import Item
from academicdigest.pipeline.normalize import canonical_url
from academicdigest.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 20, 6, 0, tzinfo=UTC)


@pytest.fixture
def scoring_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "scoring.yaml").read_text())


def make_settings(**kw) -> Settings:
    defaults = dict(
        _env_file=None,
        database_url="sqlite://",
        sources_file=ROOT / "config" / "sources.yaml",
        scoring_file=ROOT / "config" / "scoring.yaml",
        mail_to=["me@example.com"],
        mail_from="me@example.com",
    )
    return Settings(**{**defaults, **kw})


def make_item(title: str, url: str = "", source: str = "Test", tier: int = 2, hours_ago: float = 1,
              **kw) -> Item:
    url = url or f"https://example.com/{abs(hash(title))}"
    return Item(
        id=kw.pop("id", 0), source=source, guid=kw.pop("guid", title), title=title, url=url,
        canonical_url=canonical_url(url), tier=tier,
        published_at=NOW - timedelta(hours=hours_ago), **kw,
    )


def mock_client(handler) -> httpx.Client:
    """httpx client whose requests go to `handler(request) -> httpx.Response`."""
    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeNotifier:
    def __init__(self, fail: bool = False):
        self.sent = []
        self.fail = fail

    def send(self, message):
        if self.fail:
            raise ConnectionError("smtp down")
        self.sent.append(message)
