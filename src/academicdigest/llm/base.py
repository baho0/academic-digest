"""Provider-agnostic LLM interface.

Nothing in the pipeline talks to a vendor SDK directly: it asks an `LLMClient`
for a completion. A provider is a class with `complete(LLMRequest) -> LLMResponse`
registered under a name; AD_LLM_PROVIDER picks it.

    @register_provider("myprovider")
    class MyProvider:
        def __init__(self, settings: Settings) -> None:
            self.name, self.model = "myprovider", settings.llm_model
        def complete(self, request: LLMRequest) -> LLMResponse:
            ...  # call the API, raise LLMError on failure

No provider ships yet: with AD_LLM_PROVIDER=none (the default) `build_llm`
returns None and the pipeline uses rule scoring + abstract excerpts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from ..settings import Settings


class LLMError(Exception):
    """Any provider failure (network, rate limit, refusal, unparsable output)."""


class LLMBudgetExceeded(LLMError):
    """AD_LLM_MAX_CALLS reached for this run."""


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    json: bool = False  # ask the provider for a JSON object (if it supports a JSON mode)
    purpose: str = ""  # free-form tag for the usage log: "summary", "rating"...

    def cache_fields(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "system": self.system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "json": self.json,
        }


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class LLMClient(Protocol):
    name: str  # provider name, e.g. "anthropic"
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


_PROVIDERS: dict[str, Callable[[Settings], LLMClient]] = {}


def register_provider(name: str) -> Callable[[Any], Any]:
    def deco(factory: Any) -> Any:
        _PROVIDERS[name] = factory
        return factory

    return deco


def providers() -> list[str]:
    return sorted(_PROVIDERS)


def build_provider(settings: Settings) -> LLMClient | None:
    """The raw provider client, or None when LLM features are disabled."""
    name = (settings.llm_provider or "none").lower()
    if name in ("", "none", "off"):
        return None
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider {name!r} (AD_LLM_PROVIDER); registered: {providers() or 'none yet'}. "
            "See academicdigest/llm/base.py for how to add one."
        ) from None
    return factory(settings)


class BudgetedLLM:
    """Stops a run from making more than `max_calls` provider calls.

    Sits between the cache and the provider, so cache hits never count.
    """

    def __init__(self, inner: LLMClient, max_calls: int) -> None:
        self.inner = inner
        self.name, self.model = inner.name, inner.model
        self.max_calls = max_calls
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.calls >= self.max_calls:
            raise LLMBudgetExceeded(f"LLM call budget of {self.max_calls} reached")
        self.calls += 1
        return self.inner.complete(request)
