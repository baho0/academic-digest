"""Command line entry point: `academicdigest --help`."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import typer
from sqlalchemy.orm import Session

from . import db
from .delivery.smtp import build_notifier
from .llm import LLMClient, LLMError, LLMRequest, build_llm
from .llm.cache import usage
from .models import Digest
from .pipeline import run as pipeline
from .render.renderer import Rendered
from .settings import Settings, get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Daily research paper digest.")


def _setup(verbose: bool) -> Settings:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.INFO if verbose else logging.WARNING)
    return get_settings()


def _session(settings: Settings) -> Session:
    return Session(db.make_engine(settings.database_url))


def _llm(settings: Settings, session: Session) -> LLMClient | None:
    """Only build a client when the config actually uses one."""
    cfg = settings.scoring
    if cfg.get("engine", "rules") == "rules" and cfg.get("summarizer", "excerpt") == "excerpt":
        return None
    return build_llm(settings, session)


def _print_collect(results: list[pipeline.CollectResult]) -> None:
    for r in results:
        if r.error:
            status = f"ERROR {r.error}"
        elif r.source.startswith("+"):
            status = f"{r.fetched:4d} to enrich  {r.updated:4d} enriched"
        else:
            status = f"{r.fetched:4d} fetched  {r.new:4d} new  {r.updated:4d} updated"
        typer.echo(f"  {r.source:28} {status}  ({r.seconds:.1f}s)")


@app.command()
def collect(
    source: Optional[list[str]] = typer.Option(None, "--source", "-s", help="Only these source names."),
    enrich: bool = typer.Option(True, help="Run enrichers (arXiv metadata, ...) afterwards."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch all sources, store new papers and enrich them (run every few hours)."""
    settings = _setup(verbose)
    with _session(settings) as session:
        results = pipeline.collect(settings, session, source, enrich_items=enrich)
    _print_collect(results)
    fetchers = [r for r in results if not r.source.startswith("+")]
    if fetchers and all(r.error for r in fetchers):
        raise typer.Exit(1)


@app.command()
def digest(
    force: bool = typer.Option(False, help="Send again even if today's digest was already sent."),
    skip_collect: bool = typer.Option(False, help="Do not fetch sources first."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build today's digest and email it (run once every morning)."""
    settings = _setup(verbose)
    notifier = build_notifier(settings)
    with _session(settings) as session:
        if not skip_collect:
            pipeline.collect(settings, session)
        status = pipeline.run_digest(
            settings, session, notifier, datetime.now(UTC), force=force, llm=_llm(settings, session)
        )
    typer.echo(f"digest: {status}")


@app.command()
def preview(
    out: Path = typer.Option(Path("preview.html"), help="Where to write the HTML."),
    explain: bool = typer.Option(True, help="Show score breakdown under every paper."),
    collect_first: bool = typer.Option(False, "--collect", help="Fetch sources first."),
    include_sent: bool = typer.Option(False, help="Do not hide papers already emailed."),
    candidates: int = typer.Option(40, help="How many top candidates to list in the terminal."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the digest WITHOUT sending or saving it; write HTML and list scores."""
    settings = _setup(verbose)
    with _session(settings) as session:
        if collect_first:
            _print_collect(pipeline.collect(settings, session))
        d, clusters = pipeline.build_digest(
            settings, session, datetime.now(UTC), exclude_sent=not include_sent,
            llm=_llm(settings, session),
        )
    message = pipeline.render_digest(settings, d, explain=explain)
    out.write_text(message.html, encoding="utf-8")
    chosen = {id(c) for s in d.sections for c in s.clusters}
    typer.echo(f"{message.subject}\nTop {candidates} of {len(clusters)} candidates (* = selected):")
    for c in clusters[:candidates]:
        mark = "*" if id(c) in chosen else " "
        typer.echo(f" {mark} {c.score:5.1f} [{c.category:8}] {c.title[:90]}")
        if explain:
            typer.echo(f"         {', '.join(map(str, c.breakdown))}  <{', '.join(c.source_names)}>")
    typer.echo(f"\nHTML written to {out.resolve()}")


@app.command("send-test")
def send_test(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Send a tiny test email to check SMTP settings."""
    settings = _setup(verbose)
    d = Digest(date=datetime.now(UTC).strftime("%Y-%m-%d"), sections=[])
    msg = pipeline.render_digest(settings, d)
    test = Rendered(subject="Academic Digest — SMTP test", html=msg.html, text=msg.text)
    build_notifier(settings).send(test)
    typer.echo(f"test email sent to {', '.join(settings.mail_to)} via {settings.smtp_host}:{settings.smtp_port}")


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Default: AD_WEB_HOST."),
    port: Optional[int] = typer.Option(None, help="Default: AD_WEB_PORT."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the web UI + JSON API + Atom feed."""
    import uvicorn

    from .web import create_app

    settings = _setup(verbose)
    uvicorn.run(
        create_app(settings),
        host=host or settings.web_host,
        port=port or settings.web_port,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
        log_level="debug" if verbose else "info",
    )


@app.command("llm-check")
def llm_check(
    prompt: str = typer.Argument("Reply with the single word: ok"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Send one prompt to the configured LLM provider (no cache) to test the setup."""
    settings = _setup(verbose)
    try:
        client = build_llm(settings)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    if client is None:
        typer.echo("LLM disabled (AD_LLM_PROVIDER=none).")
        raise typer.Exit(1)
    try:
        r = client.complete(LLMRequest(prompt=prompt, max_tokens=50, purpose="check"))
    except LLMError as exc:
        typer.echo(f"{client.name} failed: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"{client.name}/{r.model or client.model}: {r.text!r} ({r.input_tokens} in / {r.output_tokens} out)")


@app.command("llm-usage")
def llm_usage(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Token usage of all cached LLM calls, per provider/model/purpose."""
    settings = _setup(verbose)
    with _session(settings) as session:
        rows = usage(session)
    if not rows:
        typer.echo("no LLM calls recorded")
    for provider, model, purpose, calls, tin, tout in rows:
        typer.echo(f"  {provider}/{model:30} {purpose or '-':10} {calls:6d} calls {tin:10d} in {tout:10d} out")


if __name__ == "__main__":
    app()
