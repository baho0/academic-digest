"""Paper sources. Importing this package registers every built-in source type."""

from . import arxiv, hackernews, hf_papers, rss  # noqa: F401
from .base import Source, build_source, register

__all__ = ["Source", "build_source", "register"]
