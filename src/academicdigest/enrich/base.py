"""Enricher protocol and registry.

Sources say *that* a paper appeared; enrichers look up more about papers
already stored (arXiv comments, citation counts...). They run after every
collect, over the items inside the clustering window.

Adding an enricher = subclass `Enricher`, decorate with `@register("type")`,
then list it under `enrichers:` in config/sources.yaml.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

import httpx

from ..db import ItemPatch
from ..models import Item
from ..settings import EnricherConfig, Settings

_REGISTRY: dict[str, type[Enricher]] = {}


def register(type_name: str) -> Callable[[type[Enricher]], type[Enricher]]:
    def deco(cls: type[Enricher]) -> type[Enricher]:
        cls.type_name = type_name
        _REGISTRY[type_name] = cls
        return cls

    return deco


def build_enricher(config: EnricherConfig, client: httpx.Client, settings: Settings) -> Enricher:
    try:
        cls = _REGISTRY[config.type]
    except KeyError:
        raise ValueError(
            f"Unknown enricher type {config.type!r} for {config.name!r}; known: {sorted(_REGISTRY)}"
        ) from None
    return cls(config, client, settings)


class Enricher:
    type_name: ClassVar[str] = ""

    def __init__(self, config: EnricherConfig, client: httpx.Client, settings: Settings) -> None:
        self.config = config
        self.client = client
        self.settings = settings

    @property
    def name(self) -> str:
        return self.config.name

    def opt(self, key: str, default: Any = None) -> Any:
        return self.config.options.get(key, default)

    def wants(self, item: Item) -> bool:  # pragma: no cover - interface
        """Does this item still need enriching?"""
        raise NotImplementedError

    def enrich(self, items: list[Item]) -> dict[int, ItemPatch]:  # pragma: no cover - interface
        """Look items up; return patches keyed by item id. Must not touch the DB."""
        raise NotImplementedError


def chunks(seq: list[Any], size: int) -> list[list[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]
