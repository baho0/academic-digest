"""FastAPI app: HTML pages, JSON API (/api, docs at /api/docs) and an Atom feed.

Read-only over the same database the CLI writes. Run with `academicdigest serve`
behind a reverse proxy; set AD_WEB_USER / AD_WEB_PASSWORD for basic auth.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .. import __version__, db
from ..pipeline.run import category_titles
from ..render.renderer import make_env
from ..settings import Settings, get_settings
from . import queries

_HERE = Path(__file__).parent


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session


_basic = HTTPBasic(auto_error=False)


def require_auth(
    request: Request, credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic)]
) -> None:
    settings: Settings = request.app.state.settings
    if not (settings.web_user and settings.web_password):
        return
    ok = credentials is not None and secrets.compare_digest(
        credentials.username.encode(), settings.web_user.encode()
    ) & secrets.compare_digest(credentials.password.encode(), settings.web_password.encode())
    if not ok:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "authentication required",
            headers={"WWW-Authenticate": 'Basic realm="Academic Digest"'},
        )


SessionDep = Annotated[Session, Depends(get_session)]


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or db.make_engine(settings.database_url)
    templates = Jinja2Templates(env=make_env(_HERE / "templates"))
    cats = category_titles(settings)
    top_title = str(settings.scoring.get("selection", {}).get("top_title", "Today's highlights"))

    def base_url(request: Request) -> str:
        return (settings.web_base_url or str(request.base_url)).rstrip("/")

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"categories": cats, "version": __version__, **ctx})

    app = FastAPI(
        title="Academic Digest",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.engine = engine
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")

    @app.get("/healthz", include_in_schema=False)
    def healthz(session: SessionDep) -> dict[str, Any]:
        return {"ok": True, "latest_digest": queries.latest_digest_date(session)}

    # ------------------------------------------------------------------ HTML
    html = APIRouter(dependencies=[Depends(require_auth)], include_in_schema=False)

    @html.get("/", response_class=HTMLResponse)
    def index(request: Request, session: SessionDep) -> Response:
        latest = queries.latest_digest_date(session)
        if latest is None:
            return page(request, "empty.html")
        return RedirectResponse(request.url_for("digest_page", date=latest), status_code=302)

    @html.get("/digests", response_class=HTMLResponse)
    def archive(request: Request, session: SessionDep, offset: int = 0) -> Response:
        rows = queries.list_digests(session, limit=60, offset=offset)
        return page(request, "archive.html", digests=rows, offset=offset)

    @html.get("/digests/{date}", response_class=HTMLResponse, name="digest_page")
    def digest_page(request: Request, session: SessionDep, date: str, all: bool = False) -> Response:
        d = queries.digest_detail(session, date, cats, top_title, include_all=all)
        if d is None:
            raise HTTPException(404, f"no digest for {date}")
        return page(request, "digest.html", d=d, show_all=all)

    @html.get("/digests/{date}/email", response_class=HTMLResponse)
    def digest_email(session: SessionDep, date: str) -> Response:
        body = queries.digest_email_html(session, date)
        if body is None:
            raise HTTPException(404, f"no digest for {date}")
        return HTMLResponse(body)

    @html.get("/papers", response_class=HTMLResponse)
    def papers_page(
        request: Request,
        session: SessionDep,
        q: str = "",
        days: int = 30,
        source: str | None = None,
        in_digest: bool = False,
        offset: int = 0,
    ) -> Response:
        rows = queries.search_papers(session, q, days or None, source or None, in_digest, 50, offset)
        sources = [s["source"] for s in queries.source_stats(session)]
        return page(
            request, "papers.html", papers=rows, q=q, days=days, source=source or "",
            in_digest=in_digest, offset=offset, sources=sources,
        )

    @html.get("/papers/{item_id}", response_class=HTMLResponse)
    def paper_page(request: Request, session: SessionDep, item_id: int) -> Response:
        p = queries.paper_detail(session, item_id)
        if p is None:
            raise HTTPException(404, "no such paper")
        return page(request, "paper.html", p=p)

    @html.get("/status", response_class=HTMLResponse)
    def status_page(request: Request, session: SessionDep) -> Response:
        return page(
            request, "status.html",
            stats=queries.source_stats(session),
            digests=queries.list_digests(session, limit=14),
            llm=settings.llm_provider,
        )

    @html.get("/feed.xml")
    def feed(request: Request, session: SessionDep) -> Response:
        rows = queries.feed_digests(session)
        updated = max((r.sent_at or r.created_at for r in rows), default=None)
        body = templates.get_template("feed.xml").render(
            digests=rows,
            base=base_url(request),
            updated=(db._aware(updated) or datetime.now(UTC)).isoformat(),
            aware=db._aware,
        )
        return Response(body, media_type="application/atom+xml")

    # ------------------------------------------------------------------ JSON
    api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)], tags=["api"])

    @api.get("/digests")
    def api_digests(
        session: SessionDep,
        limit: Annotated[int, Query(ge=1, le=365)] = 30,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        """Digest archive, newest first."""
        return queries.list_digests(session, limit, offset)

    @api.get("/digests/latest")
    def api_latest(session: SessionDep, all: bool = False) -> dict[str, Any]:
        """The most recent digest (see /api/digests/{date})."""
        latest = queries.latest_digest_date(session)
        if latest is None:
            raise HTTPException(404, "no digests yet")
        return queries.digest_detail(session, latest, cats, top_title, include_all=all)  # type: ignore[return-value]

    @api.get("/digests/{date}")
    def api_digest(session: SessionDep, date: str, all: bool = False) -> dict[str, Any]:
        """One digest: sections with selected papers; `all=true` adds every candidate with its score breakdown."""
        d = queries.digest_detail(session, date, cats, top_title, include_all=all)
        if d is None:
            raise HTTPException(404, f"no digest for {date}")
        return d

    @api.get("/papers")
    def api_papers(
        session: SessionDep,
        q: str = "",
        days: Annotated[int, Query(ge=0, description="0 = all time")] = 30,
        source: str | None = None,
        in_digest: bool = False,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        """Search collected papers by title, abstract, author or arXiv id."""
        return queries.search_papers(session, q, days or None, source, in_digest, limit, offset)

    @api.get("/papers/{item_id}")
    def api_paper(session: SessionDep, item_id: int) -> dict[str, Any]:
        """One paper with every source that reported it and the digests it appeared in."""
        p = queries.paper_detail(session, item_id)
        if p is None:
            raise HTTPException(404, "no such paper")
        return p

    @api.get("/status")
    def api_status(session: SessionDep) -> dict[str, Any]:
        """Per-source collection stats and recent digest statuses (for monitoring)."""
        return {
            "version": __version__,
            "sources": queries.source_stats(session),
            "digests": queries.list_digests(session, limit=14),
            "llm_provider": settings.llm_provider,
        }

    app.include_router(html)
    app.include_router(api)
    return app
