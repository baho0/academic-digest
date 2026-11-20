"""Scoring: how worth reading is each paper?

`Scorer` is the extension point (select with `engine:` in config/scoring.yaml):
  rules   RuleScorer — transparent, every point explained
  hybrid  rules first, then an LLM rates the top `llm.rerank_top_k` papers
          against your research profile and nudges their score
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Any, Protocol

from ..llm import LLMBudgetExceeded, LLMClient, LLMError, LLMRequest, parse_json, render_prompt
from ..models import Cluster, ScoreLine
from .normalize import domain, fold_name

log = logging.getLogger(__name__)

FIELDS = ("title", "abstract", "venue", "any")


class Scorer(Protocol):
    def score(self, clusters: list[Cluster], now: datetime) -> None:
        """Set `score` and `breakdown` on every cluster (in place)."""


class RuleScorer:
    def __init__(self, cfg: dict[str, Any], source_weights: dict[str, float] | None = None) -> None:
        w = cfg.get("weights", {})
        self.source_w = source_weights or {}
        self.off_topic = float(cfg.get("off_topic_penalty", 0.0))
        self.tier_w = {int(k): float(v) for k, v in w.get("tier", {}).items()}
        self.cov_w = float(w.get("coverage_per_source", 2.0))
        self.cov_max = int(w.get("coverage_max_sources", 3))
        self.cov_tier = int(w.get("coverage_max_tier", 2))
        self.eng_w = {k: float(v) for k, v in w.get("engagement", {}).items()}
        self.code_w = float(w.get("code_available", 0.0))
        fresh = w.get("freshness", {})
        self.grace = float(fresh.get("grace_hours", 24))
        self.half_life = float(fresh.get("half_life_hours", 72))
        self.boost_cap = float(cfg.get("boost_cap", 5.0))
        self.boosts = _compile(cfg.get("boost_patterns", []))
        self.penalties = _compile(cfg.get("penalty_patterns", []))
        interests = cfg.get("interests") or {}
        self.interest_cap = float(interests.get("cap", 4.0))
        self.interests = _compile(interests.get("patterns", []))
        authors = cfg.get("watch_authors") or {}
        self.author_w = float(authors.get("weight", 3.0))
        self.authors = {fold_name(n): n for n in authors.get("names", [])}
        self.domain_penalty = float(cfg.get("domain_penalty", -8.0))
        self.blocked = [d.lower() for d in cfg.get("blocked_domains", [])]

    def score(self, clusters: list[Cluster], now: datetime) -> None:
        for c in clusters:
            lines = self.explain(c, now)
            c.breakdown = lines
            c.score = sum(line.delta for line in lines)

    def explain(self, c: Cluster, now: datetime) -> list[ScoreLine]:
        lines: list[ScoreLine] = []
        best_tier = min(i.tier for i in c.items)
        base = self.tier_w.get(best_tier, 0.0)
        label = f"tier{best_tier}"
        for name in c.source_names:
            if name in self.source_w and self.source_w[name] > base:
                base, label = self.source_w[name], name
        if base:
            lines.append(ScoreLine(label, base))

        # Firehose sources (arXiv lists every paper) are not an endorsement.
        endorsing = {i.source for i in c.items if i.tier <= self.cov_tier}
        extra_sources = min(len(endorsing) - 1, self.cov_max)
        if extra_sources > 0:
            lines.append(ScoreLine(f"coverage x{extra_sources + 1}", extra_sources * self.cov_w))

        for metric, weight in self.eng_w.items():
            value = c.metric(metric)
            if value > 0:
                lines.append(ScoreLine(f"{metric}={int(value)}", weight * math.log10(value + 1)))

        if self.code_w and c.code_url:
            lines.append(ScoreLine("code", self.code_w))

        texts = {
            "title": c.title,
            "abstract": c.abstract,
            "venue": " ".join(filter(None, [c.extra("comment"), c.extra("journal_ref")])),
        }
        texts["any"] = " ".join(texts.values())

        boost = 0.0
        for label, field, pat, weight in self.boosts:
            if pat.search(texts[field]):
                delta = min(weight, self.boost_cap - boost)
                if delta > 0:
                    lines.append(ScoreLine(label, delta))
                    boost += delta
        for label, field, pat, weight in self.penalties:
            if pat.search(texts[field]):
                lines.append(ScoreLine(label, weight))

        interest = 0.0
        for label, field, pat, weight in self.interests:
            if field == "title":  # default: full weight in the title, half in the abstract
                hit = weight if pat.search(c.title) else weight / 2 if pat.search(c.abstract) else 0.0
            else:
                hit = weight if pat.search(texts[field]) else 0.0
            delta = min(hit, self.interest_cap - interest)
            if delta > 0:
                lines.append(ScoreLine(f"interest:{label}", delta))
                interest += delta

        for author in c.authors:
            if fold_name(author) in self.authors:
                lines.append(ScoreLine(f"author:{author}", self.author_w))
                break

        if c.topic_hits <= 0 and self.off_topic:
            lines.append(ScoreLine("off-topic", self.off_topic))

        dom = domain(c.url)
        if any(dom == b or dom.endswith("." + b) for b in self.blocked):
            lines.append(ScoreLine(f"blocked:{dom}", self.domain_penalty))

        subtotal = sum(line.delta for line in lines)
        newest = c.newest
        if newest and subtotal > 0:
            age_h = max(0.0, (now - newest).total_seconds() / 3600)
            if age_h > self.grace:
                factor = 0.5 ** ((age_h - self.grace) / self.half_life)
                lines.append(ScoreLine(f"age {age_h:.0f}h", subtotal * factor - subtotal))
        return lines


class HybridScorer:
    """Rule scores for everyone; an LLM rating (0-10) for the top candidates.

    The rating moves the score by (rating - center) * weight, so a 9/10 with
    weight 1 adds +4 and a 2/10 subtracts 3. Failures leave the rule score as is.
    """

    def __init__(self, rules: RuleScorer, llm: LLMClient, cfg: dict[str, Any]) -> None:
        llm_cfg = cfg.get("llm", {}) or {}
        self.rules = rules
        self.llm = llm
        self.top_k = int(llm_cfg.get("rerank_top_k", 30))
        self.weight = float(llm_cfg.get("weight", 1.0))
        self.center = float(llm_cfg.get("center", 5.0))
        self.profile = str(llm_cfg.get("profile", "")).strip() or "General computer science researcher."

    def rate(self, c: Cluster) -> tuple[float, str]:
        system, prompt = render_prompt("rate", paper=c, profile=self.profile)
        response = self.llm.complete(
            LLMRequest(prompt=prompt, system=system, max_tokens=200, json=True, purpose="rating")
        )
        data = parse_json(response.text)
        try:
            rating = max(0.0, min(10.0, float(data["score"])))
        except (KeyError, TypeError, ValueError):
            raise LLMError(f"rating missing in {data!r}") from None
        return rating, str(data.get("reason", ""))

    def score(self, clusters: list[Cluster], now: datetime) -> None:
        self.rules.score(clusters, now)
        for c in sorted(clusters, key=lambda c: c.score, reverse=True)[: self.top_k]:
            try:
                rating, reason = self.rate(c)
            except LLMBudgetExceeded as exc:
                log.warning("%s; remaining papers keep their rule score", exc)
                break
            except LLMError as exc:
                log.warning("LLM rating failed for %r: %s", c.title[:60], exc)
                continue
            delta = (rating - self.center) * self.weight
            c.breakdown.append(ScoreLine(f"llm {rating:g}/10", delta))
            c.score += delta
            if reason:
                c.notes["llm_reason"] = reason


def _compile(specs: list[dict[str, Any]]) -> list[tuple[str, str, re.Pattern[str], float]]:
    out = []
    for s in specs:
        field = s.get("field", "title")
        if field not in FIELDS:
            raise ValueError(f"pattern {s.get('label', s['pattern'])!r}: field must be one of {FIELDS}")
        flags = 0 if s.get("case") == "sensitive" else re.I
        out.append((s.get("label", s["pattern"]), field, re.compile(s["pattern"], flags), float(s["weight"])))
    return out


def build_scorer(
    cfg: dict[str, Any],
    source_weights: dict[str, float] | None = None,
    llm: LLMClient | None = None,
) -> Scorer:
    engine = cfg.get("engine", "rules")
    rules = RuleScorer(cfg, source_weights)
    if engine == "rules":
        return rules
    if engine == "hybrid":
        if llm is None:
            raise ValueError("engine: hybrid needs an LLM provider (AD_LLM_PROVIDER)")
        return HybridScorer(rules, llm, cfg)
    raise ValueError(f"unknown scoring engine {engine!r} (rules | hybrid)")
