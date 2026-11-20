"""DB-backed response cache + usage log (`llm_calls` table).

Re-running `preview` or `digest` on the same candidates costs nothing: the same
model + request returns the stored answer. Token counts are kept for cost tracking.
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import LlmCallRow, _naive, utcnow
from .base import LLMClient, LLMRequest, LLMResponse


def request_key(provider: str, model: str, request: LLMRequest) -> str:
    payload = json.dumps(
        {"provider": provider, "model": model, **request.cache_fields()}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedLLM:
    def __init__(self, inner: LLMClient, session: Session) -> None:
        self.inner = inner
        self.name, self.model = inner.name, inner.model
        self.session = session

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request_key(self.name, self.model, request)
        row = self.session.scalar(select(LlmCallRow).where(LlmCallRow.key == key))
        if row is not None:
            return LLMResponse(text=row.response, model=row.model, cached=True)
        response = self.inner.complete(request)
        self.session.add(
            LlmCallRow(
                key=key,
                provider=self.name,
                model=response.model or self.model,
                purpose=request.purpose,
                response=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                created_at=_naive(utcnow()),
            )
        )
        self.session.commit()
        return response


def usage(session: Session) -> list[tuple[str, str, str, int, int, int]]:
    """(provider, model, purpose, calls, input_tokens, output_tokens) totals."""
    rows = session.execute(
        select(
            LlmCallRow.provider,
            LlmCallRow.model,
            LlmCallRow.purpose,
            func.count(),
            func.coalesce(func.sum(LlmCallRow.input_tokens), 0),
            func.coalesce(func.sum(LlmCallRow.output_tokens), 0),
        ).group_by(LlmCallRow.provider, LlmCallRow.model, LlmCallRow.purpose)
    ).all()
    return [tuple(r) for r in rows]  # type: ignore[misc]
