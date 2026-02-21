#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route A (S0-only): HLP seeds -> S0 (first-generation citing works) -> missing intercept
-> (hard gate + soft scoring) -> 2x2 coarse classification -> outputs works.csv + edges.csv.

Enhancements in this version:
- API key support via:
    (1) environment variable: OPENALEX_API_KEY
    (2) CLI: --api-key (overrides env var)
- Customizable header name: --api-key-header (default: Authorization)
- Optional Bearer wrapping: --api-key-bearer / --no-api-key-bearer

Notes:
- OpenAlex normally does NOT require an API key. If you have one or your environment
  requires it, this script can attach it to every request.
- This script does NOT perform multi-hop BFS expansion. It only constructs S0.

Outputs:
- works.csv: ALL S0 works (eligible + non-eligible + missing), with provenance, gate, 2x2, metadata flags.
- edges.csv: ALL citation edges S0 -> seeds, preserving provenance.
- runlog.json: reproducibility metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


# -------------------------
# Utilities
# -------------------------

def now_utc_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def normalize_work_id(wid: str) -> str:
    """
    Normalize an OpenAlex work id which can appear as:
      - "W123..."
      - "https://openalex.org/W123..."
      - "https://api.openalex.org/works/W123..."
    """
    if not wid:
        return ""
    wid = wid.strip()
    if wid.startswith("https://openalex.org/"):
        return wid.rstrip("/").split("/")[-1]
    if wid.startswith("https://api.openalex.org/works/"):
        return wid.rstrip("/").split("/")[-1]
    return wid


def dict_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# -------------------------
# OpenAlex Client
# -------------------------

class OpenAlexClient:
    def __init__(
        self,
        base_url: str = "https://api.openalex.org",
        mailto: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 0.6,
        backoff_jitter: float = 0.2,
        ua: str = "routeA_s0_only/1.1",
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_bearer: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.mailto = mailto
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_jitter = backoff_jitter

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": ua})

        # Attach API key to ALL requests, if provided
        if api_key:
            key_val = api_key.strip()
            if api_key_bearer:
                # If the user already passed "Bearer ...", keep it; else wrap with Bearer.
                if not key_val.lower().startswith("bearer "):
                    key_val = f"Bearer {key_val}"
            # Example:
            #   Authorization: Bearer <token>
            # Or if you want X-API-KEY: <key>, set --api-key-header X-API-KEY and disable bearer.
            self.session.headers.update({api_key_header: key_val})

    def _sleep_backoff(self, attempt: int):
        t = (self.backoff_base * (2 ** attempt)) + random.random() * self.backoff_jitter
        time.sleep(min(t, 6.0))

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url + path
        params = dict(params or {})
        if self.mailto:
            params["mailto"] = self.mailto

        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    self._sleep_backoff(attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = str(e)
                self._sleep_backoff(attempt)
        raise RuntimeError(f"OpenAlex GET failed: {url} params={params} err={last_err}")

    def resolve_work_id_from_doi(self, doi: str) -> Optional[str]:
        doi = (doi or "").strip().lower()
        if not doi:
            return None
        data = self._get("/works", params={"filter": f"doi:{doi}", "per-page": 5})
        results = data.get("results") or []
        if not results:
            return None
        # Prefer exact DOI match when possible
        for w in results:
            wdoi = (w.get("doi") or "").lower()
            if wdoi.endswith(doi):
                return normalize_work_id(w.get("id") or "")
        return normalize_work_id(results[0].get("id") or "")

    def fetch_work(self, wid: str) -> Dict[str, Any]:
        wid = normalize_work_id(wid)
        return self._get(f"/works/{wid}", params={})

    def fetch_ids_by_url(
        self,
        url: str,
        cap: int = 200000,
        per_page: int = 200,
        verbose: bool = False,
    ) -> List[str]:
        """
        Fetch OpenAlex works from an absolute URL like:
          - https://api.openalex.org/works?filter=cites:W...&cursor=*
          - cited_by_api_url
        Return list of normalized Work IDs.

        Uses cursor-based pagination (cursor=*).
        """
        if not url:
            return []

        if url.startswith("/"):
            url = self.base_url + url

        out: List[str] = []
        cursor = "*"
        page_count = 0

        while True:
            if "?" in url:
                base, q = url.split("?", 1)
                qparams: Dict[str, str] = {}
                for part in q.split("&"):
                    if not part:
                        continue
                    if "=" in part:
                        k, v = part.split("=", 1)
                        qparams[k] = v
                    else:
                        qparams[part] = ""
            else:
                base, qparams = url, {}

            qparams = dict(qparams)
            qparams["cursor"] = cursor
            qparams["per-page"] = str(per_page)
            if self.mailto:
                qparams["mailto"] = self.mailto

            # Use the same session => API key header automatically included
            try:
                r = self.session.get(base, params=qparams, timeout=self.timeout)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    page_count += 1
                    if verbose:
                        print(f"[WARN] retry HTTP {r.status_code} for {base}", file=sys.stderr)
                    time.sleep(1.0)
                    continue
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                if verbose:
                    print(f"[WARN] page fetch failed: {e}", file=sys.stderr)
                time.sleep(1.0)
                continue

            results = data.get("results") or []
            for w in results:
                out.append(normalize_work_id(w.get("id") or ""))
                if len(out) >= cap:
                    return [x for x in out if x]

            cursor = data.get("meta", {}).get("next_cursor")
            page_count += 1
            if verbose and page_count % 20 == 0:
                print(f"[INFO] fetched {len(out)} ids so far ...", file=sys.stderr)

            if not cursor:
                break

        return [x for x in out if x]


# -------------------------
# Parsing OpenAlex Work JSON
# -------------------------

def invert_abstract(inv: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstruct abstract from OpenAlex inverted index:
      {"word":[pos1,pos2,...], ...}
    """
    if not inv or not isinstance(inv, dict):
        return ""
    positions: Dict[int, str] = {}
    for word, pos_list in inv.items():
        if not isinstance(pos_list, list):
            continue
        for p in pos_list:
            if isinstance(p, int):
                positions[p] = word
    if not positions:
        return ""
    max_pos = max(positions.keys())
    tokens = [""] * (max_pos + 1)
    for p, w in positions.items():
        tokens[p] = w
    return norm_spaces(" ".join(tokens))


def top_concepts(work: Dict[str, Any], k: int = 10) -> str:
    concepts = work.get("concepts") or []
    if not isinstance(concepts, list):
        return ""
    def score(c): return c.get("score") if isinstance(c, dict) else 0.0
    concepts_sorted = sorted([c for c in concepts if isinstance(c, dict)], key=score, reverse=True)
    names = [c.get("display_name") for c in concepts_sorted[:k] if c.get("display_name")]
    return norm_spaces(" | ".join(names))


def top_topics(work: Dict[str, Any], k: int = 10) -> str:
    topics = work.get("topics") or []
    if not isinstance(topics, list):
        return ""
    def score(t): return t.get("score") if isinstance(t, dict) else 0.0
    topics_sorted = sorted([t for t in topics if isinstance(t, dict)], key=score, reverse=True)
    names = [t.get("display_name") for t in topics_sorted[:k] if t.get("display_name")]
    return norm_spaces(" | ".join(names))


def parse_authors(work: Dict[str, Any], cap: int = 20) -> str:
    auths = dict_get(work, ["authorships"], []) or []
    out: List[str] = []
    for a in auths[:cap]:
        if not isinstance(a, dict):
            continue
        author = a.get("author") or {}
        name = author.get("display_name")
        if name:
            out.append(name)
    return norm_spaces(" | ".join(out))


def parse_institutions(work: Dict[str, Any], cap: int = 30) -> str:
    auths = dict_get(work, ["authorships"], []) or []
    inst: List[str] = []
    for a in auths:
        insts = a.get("institutions") or []
        for i in insts:
            if isinstance(i, dict) and i.get("display_name"):
                inst.append(i["display_name"])
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in inst:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
        if len(uniq) >= cap:
            break
    return norm_spaces(" | ".join(uniq))


def parse_referenced_works(work: Dict[str, Any], cap: int = 200) -> str:
    refs = work.get("referenced_works") or []
    if not isinstance(refs, list):
        return ""
    ids = [normalize_work_id(x) for x in refs[:cap] if x]
    return norm_spaces(" | ".join([x for x in ids if x]))


def parse_work_json(work: Dict[str, Any]) -> Dict[str, Any]:
    wid = normalize_work_id(work.get("id") or "")
    title = norm_spaces(work.get("display_name") or "")
    year = work.get("publication_year") or ""
    wtype = work.get("type") or ""
    doi = work.get("doi") or ""
    venue = dict_get(work, ["host_venue", "display_name"], "") or ""
    language = work.get("language") or ""
    cited_by_count = work.get("cited_by_count") or 0

    abstract = invert_abstract(work.get("abstract_inverted_index"))
    concepts_top = top_concepts(work, k=10)
    topics_top = top_topics(work, k=10)

    authors = parse_authors(work, cap=20)
    institutions = parse_institutions(work, cap=30)

    referenced_works = parse_referenced_works(work, cap=200)
    referenced_works_count = len(work.get("referenced_works") or []) if isinstance(work.get("referenced_works"), list) else 0

    cited_by_api_url = work.get("cited_by_api_url") or ""

    return {
        "openalex_id": wid,
        "doi": doi,
        "title": title,
        "publication_year": year,
        "type": wtype,
        "venue": venue,
        "language": language,
        "authors": authors,
        "institutions": institutions,
        "cited_by_count": cited_by_count,
        "referenced_works_count": referenced_works_count,
        "referenced_works": referenced_works,
        "cited_by_api_url": cited_by_api_url,
        "abstract": abstract,
        "has_abstract": 1 if abstract else 0,
        "concepts_top": concepts_top,
        "topics_top": topics_top,
    }


# -------------------------
# Patterns: Gate + 2x2
# -------------------------

PATTERNS_OPTION = [
    r"\boption(s)?\b",
    r"\bderivative(s)?\b",
    r"\bcall(s)?\b",
    r"\bput(s)?\b",
    r"\boption\s+pricing\b",
    r"\bderivative\s+pricing\b",
    r"\bimplied\s+vol(atility)?\b",
    r"\biv(s)?\b",
    r"\bvol(atility)?\s+surface\b",
    r"\bvol(atility)?\s+smile\b",
    r"\bsmile\b",
    r"\bskew\b",
    r"\barbitrage[-\s]*free\b",
    r"\bcalibrat(e|ion|ed|ing)\b",
    r"\bgreeks\b",
    r"\bdelta\b",
    r"\bgamma\b",
    r"\bvega\b",
    r"\btheta\b",
]

PATTERNS_ML = [
    r"\bmachine\s+learning\b",
    r"\bstatistical\s+learning\b",
    r"\bnonparametric\b",
    r"\bneural\s+network(s)?\b",
    r"\bartificial\s+neural\s+network(s)?\b",
    r"\bann(s)?\b",
    r"\bdeep\s+learning\b",
    r"\bdeep\s+neural\b",
    r"\bmlp\b",
    r"\bfeed[-\s]*forward\b",
    r"\brnn(s)?\b",
    r"\blstm(s)?\b",
    r"\bgru(s)?\b",
    r"\btransformer(s)?\b",
    r"\bsupport\s+vector\s+machine(s)?\b",
    r"\bsvm(s)?\b",
    r"\bgaussian\s+process(es)?\b",
    r"\bgp\s+regression\b",
    r"\bkernel\s+regression\b",
    r"\bkernel(s)?\b",
    r"\brandom\s+forest(s)?\b",
    r"\bgradient\s+boost(ing)?\b",
    r"\bxgboost\b",
    r"\blightgbm\b",
    r"\breinforcement\s+learning\b",
    r"\brl\b",
    r"\b(normalizing\s+)?flow(s)?\b",
    r"\bvariational\b",
    r"\bvae\b",
    r"\bgan(s)?\b",
    r"\bencoder\b",
    r"\bdecoder\b",
    r"\bapproximation\b",
]

PATTERNS_EXCLUDE = [
    r"\bportfolio\b",
    r"\basset\s+allocation\b",
    r"\bsentiment\b",
    r"\bnews\b",
    r"\btext\b",
    r"\btwitter\b",
    r"\breturn(s)?\s+predict(ion|ing|ed)\b",
    r"\bstock\s+predict(ion|ing|ed)\b",
    r"\bprice\s+predict(ion|ing|ed)\b",
    r"\balgorithmic\s+trading\b",
    r"\bhigh\s+frequency\b",
]

PATTERNS_DYNAMIC = [
    r"\btime\s+evolution\b",
    r"\btime[-\s]*varying\b",
    r"\btemporal\b",
    r"\bdynamic(s)?\b",
    r"\btime\s+series\b",
    r"\bsequential\b",
    r"\bsequence\b",
    r"\brecurrent\b",
    r"\brnn\b",
    r"\blstm\b",
    r"\bgru\b",
    r"\btransformer\b",
    r"\bonline\s+learning\b",
    r"\badaptive\b",
    r"\bstate[-\s]*space\b",
    r"\bkalman\b",
    r"\bhidden\s+markov\b",
    r"\bregime\s+switch(ing)?\b",
    r"\bsde\b",
    r"\bstochastic\s+process(es)?\b",
    r"\bneural\s+sde\b",
    r"\bevolution\b",
]

PATTERNS_IV = [
    r"\bimplied\s+vol(atility)?\b",
    r"\biv\b",
    r"\bvol(atility)?\s+surface\b",
    r"\bvol(atility)?\s+smile\b",
    r"\bsmile\b",
    r"\bskew\b",
    r"\bsvi\b",
    r"\bsabr\b",
    r"\barbitrage[-\s]*free\b",
    r"\bcalibrat(e|ion|ed|ing)\b",
]

PATTERNS_PRICE = [
    r"\boption\s+price(s)?\b",
    r"\boption\s+pricing\b",
    r"\bderivative\s+pricing\b",
    r"\bpricing\s+formula\b",
    r"\bprice(s)?\b",
    r"\bpayoff\b",
    r"\bvaluation\b",
]


def compile_patterns(pats: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in pats]


CP_OPTION = compile_patterns(PATTERNS_OPTION)
CP_ML = compile_patterns(PATTERNS_ML)
CP_EX = compile_patterns(PATTERNS_EXCLUDE)
CP_DYN = compile_patterns(PATTERNS_DYNAMIC)
CP_IV = compile_patterns(PATTERNS_IV)
CP_PRICE = compile_patterns(PATTERNS_PRICE)


def count_hits(text: str, patterns: List[re.Pattern]) -> int:
    if not text:
        return 0
    c = 0
    for p in patterns:
        if p.search(text):
            c += 1
    return c


def list_hits(text: str, patterns: List[re.Pattern], cap: int = 8) -> List[str]:
    if not text:
        return []
    hits: List[str] = []
    for p in patterns:
        if p.search(text):
            hits.append(p.pattern)
        if len(hits) >= cap:
            break
    return hits


# -------------------------
# Gate: hard gate + soft scoring
# -------------------------

@dataclass
class GateConfig:
    w_title: float = 4.0
    w_abs: float = 3.0
    w_concept: float = 1.0
    bonus_iv_surface: float = 0.0   # marked: default off, not deleted
    penalty_exclude: float = 4.0
    threshold_high: float = 10.0
    threshold_mid: float = 6.0


class Gate:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg

    def score_and_tier(
        self,
        title: str,
        abstract: str,
        concepts_top: str,
        topics_top: str,
    ) -> Tuple[float, str, str]:
        """
        Return (score, tier, reason), tier in {HIGH, MID, LOW, EXCLUDE}.
        """
        title_t = title or ""
        abs_t = abstract or ""
        concept_t = norm_spaces((concepts_top or "") + " " + (topics_top or ""))

        opt_t = count_hits(title_t, CP_OPTION)
        opt_a = count_hits(abs_t, CP_OPTION)
        opt_c = count_hits(concept_t, CP_OPTION)

        ml_t = count_hits(title_t, CP_ML)
        ml_a = count_hits(abs_t, CP_ML)
        ml_c = count_hits(concept_t, CP_ML)

        ex_t = count_hits(title_t, CP_EX)
        ex_a = count_hits(abs_t, CP_EX)
        ex_total = ex_t + ex_a

        opt_total = opt_t + opt_a + opt_c
        ml_total = ml_t + ml_a + ml_c

        hard_pass = (opt_total >= 1) and (ml_total >= 1)

        score = (
            self.cfg.w_title * (opt_t + ml_t)
            + self.cfg.w_abs * (opt_a + ml_a)
            + self.cfg.w_concept * (opt_c + ml_c)
        )

        iv_like = count_hits(norm_spaces(title_t + " " + abs_t), CP_IV)
        if iv_like >= 1:
            score += self.cfg.bonus_iv_surface

        if ex_total > 0 and opt_total <= 1:
            score -= self.cfg.penalty_exclude * ex_total

        if ex_total >= 2 and opt_total == 0:
            tier = "EXCLUDE"
        else:
            if not hard_pass:
                tier = "LOW"
            else:
                if score >= self.cfg.threshold_high:
                    tier = "HIGH"
                elif score >= self.cfg.threshold_mid:
                    tier = "MID"
                else:
                    tier = "LOW"

        reason = (
            f"hard={'1' if hard_pass else '0'}"
            f"|opt={opt_total}(t{opt_t},a{opt_a},c{opt_c})"
            f"|ml={ml_total}(t{ml_t},a{ml_a},c{ml_c})"
            f"|ex={ex_total}(t{ex_t},a{ex_a})"
            f"|iv_like={iv_like}"
        )
        return score, tier, reason


# -------------------------
# 2x2 coarse classification
# -------------------------

def normalize_for_match(*parts: str) -> str:
    text = " ".join([p for p in parts if p])
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return norm_spaces(text)


def classify_2x2(
    title: str,
    abstract: str,
    concepts_top: str,
    topics_top: str,
) -> Tuple[str, str, str, str, str]:
    """
    Return:
      axis_temporal: static/dynamic/uncertain
      axis_target: price/iv/both/uncertain
      class_2x2: SP/SI/DP/DI/UNK
      class_confidence: HIGH/MID/LOW
      class_evidence: short evidence string
    """
    text_all = normalize_for_match(title or "", abstract or "", concepts_top or "", topics_top or "")

    dyn_hits = count_hits(text_all, CP_DYN)
    iv_hits = count_hits(text_all, CP_IV)
    pr_hits = count_hits(text_all, CP_PRICE)

    axis_temporal = "dynamic" if dyn_hits >= 1 else "static"
    if not text_all:
        axis_temporal = "uncertain"

    if iv_hits >= 1 and pr_hits == 0:
        axis_target = "iv"
    elif pr_hits >= 1 and iv_hits == 0:
        axis_target = "price"
    elif pr_hits >= 1 and iv_hits >= 1:
        axis_target = "both"
    else:
        axis_target = "uncertain"

    if axis_temporal in ("static", "dynamic") and axis_target in ("price", "iv"):
        if axis_temporal == "static" and axis_target == "price":
            cls = "SP"
        elif axis_temporal == "static" and axis_target == "iv":
            cls = "SI"
        elif axis_temporal == "dynamic" and axis_target == "price":
            cls = "DP"
        else:
            cls = "DI"
    else:
        cls = "UNK"

    axes_clear = int(axis_temporal in ("static", "dynamic")) + int(axis_target in ("price", "iv"))
    if axes_clear == 2:
        conf = "HIGH"
    elif axes_clear == 1:
        conf = "MID"
    else:
        conf = "LOW"

    ev_dyn = list_hits(text_all, CP_DYN, cap=3)
    ev_iv = list_hits(text_all, CP_IV, cap=3)
    ev_pr = list_hits(text_all, CP_PRICE, cap=3)
    ev: List[str] = []
    if ev_dyn:
        ev.append("dyn:" + ",".join(ev_dyn))
    if ev_iv:
        ev.append("iv:" + ",".join(ev_iv))
    if ev_pr:
        ev.append("price:" + ",".join(ev_pr))
    evidence = norm_spaces(" | ".join(ev))

    return axis_temporal, axis_target, cls, conf, evidence


# -------------------------
# Missing intercept (marked)
# -------------------------

def severe_missing_check(parsed: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Marked rule:
      severe_missing=True iff title=="" AND abstract=="" AND concepts_top=="" AND topics_top==""
    Returns (severe_missing, metadata_status, metadata_reason)
    """
    title = parsed.get("title") or ""
    abstract = parsed.get("abstract") or ""
    concepts_top = parsed.get("concepts_top") or ""
    topics_top = parsed.get("topics_top") or ""

    no_title = (not title)
    no_abs = (not abstract)
    no_concepts = (not concepts_top)
    no_topics = (not topics_top)

    severe = no_title and no_abs and no_concepts and no_topics

    if severe:
        return True, "MISSING_SEVERE", "no_title|no_abstract|no_concepts|no_topics"
    return False, "OK", ""


# -------------------------
# CSV writing
# -------------------------

WORKS_HEADER = [
    "openalex_id", "doi", "title", "publication_year", "type", "venue", "language",
    "authors", "institutions",
    "cited_by_count", "referenced_works_count", "referenced_works", "cited_by_api_url",
    "has_abstract", "abstract", "concepts_top", "topics_top",
    "route_sources", "first_seen_hop", "first_seen_from",
    "metadata_status", "metadata_reason",
    "relevance_score", "relevance_tier", "relevance_reason", "is_eligible",
    "axis_temporal", "axis_target", "class_2x2", "class_confidence", "class_evidence",
]

EDGES_HEADER = [
    "src_openalex_id", "dst_openalex_id", "edge_type", "hop_generated", "from_seed"
]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -------------------------
# Main pipeline (S0-only)
# -------------------------

def run(args: argparse.Namespace) -> None:
    # Resolve API key from CLI first, else env var
    api_key = args.api_key or os.environ.get("OPENALEX_API_KEY") or None

    client = OpenAlexClient(
        base_url=args.base_url,
        mailto=args.mailto,
        timeout=args.timeout,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        backoff_jitter=args.backoff_jitter,
        ua=args.user_agent,
        api_key=api_key,
        api_key_header=args.api_key_header,
        api_key_bearer=args.api_key_bearer,
    )

    out_dir = args.out_dir
    ensure_dir(out_dir)

    # -------- Block I: Seeds resolve + cited-by + S0 + provenance --------
    seed_inputs: List[Tuple[str, str]] = []
    for s in args.seed:
        if "=" not in s:
            raise ValueError(f"Seed must be SEEDKEY=DOI, got: {s}")
        seed_key, doi = s.split("=", 1)
        seed_inputs.append((seed_key.strip(), doi.strip()))

    seeds: List[Dict[str, Any]] = []
    for seed_key, doi in seed_inputs:
        wid = client.resolve_work_id_from_doi(doi)
        if not wid:
            print(f"[WARN] could not resolve DOI -> Work ID: {seed_key} doi={doi}", file=sys.stderr)
            continue
        seeds.append({"seed_key": seed_key, "doi": doi, "openalex_id": wid})

    if not seeds:
        raise RuntimeError("No seeds were resolved. Check DOI inputs.")

    prov_map: Dict[str, Set[str]] = {}
    seed_meta: List[Dict[str, Any]] = []

    for s in seeds:
        seed_wid = s["openalex_id"]
        raw_seed = client.fetch_work(seed_wid)

        cited_by_url = raw_seed.get("cited_by_api_url") or ""
        has_cited_by_api_url = bool(cited_by_url)
        if not cited_by_url:
            cited_by_url = f"{client.base_url}/works?filter=cites:{seed_wid}"

        citers = client.fetch_ids_by_url(
            cited_by_url,
            cap=args.forward_cap_seed,
            per_page=args.per_page,
            verbose=args.verbose_fetch,
        )

        for x in citers:
            if x:
                prov_map.setdefault(x, set()).add(s["seed_key"])

        seed_meta.append({
            "seed_key": s["seed_key"],
            "doi": s["doi"],
            "openalex_id": seed_wid,
            "has_cited_by_api_url": has_cited_by_api_url,
            "cited_by_url_used": cited_by_url,
            "cited_by_count_retrieved": len(citers),
        })

        print(f"[INFO] Seed {s['seed_key']} ({seed_wid}): citers={len(citers)} (has_cited_by_api_url={has_cited_by_api_url})")

    s0_ids = sorted(prov_map.keys())
    print(f"[INFO] S0 size (union of citers across seeds): {len(s0_ids)}")

    # -------- Block II/III/IV: Missing intercept -> gate -> 2x2 -> outputs --------
    gate_cfg = GateConfig(
        w_title=args.w_title,
        w_abs=args.w_abs,
        w_concept=args.w_concept,
        bonus_iv_surface=args.bonus_iv_surface,   # default 0
        penalty_exclude=args.penalty_exclude,
        threshold_high=args.threshold_high,
        threshold_mid=args.threshold_mid,
    )
    gate = Gate(gate_cfg)

    works_path = os.path.join(out_dir, "works.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    runlog_path = os.path.join(out_dir, "runlog.json")

    # edges.csv: ALL edges S0 -> seed(s)
    with open(edges_path, "w", newline="", encoding="utf-8") as f_edge:
        w_edge = csv.DictWriter(f_edge, fieldnames=EDGES_HEADER)
        w_edge.writeheader()
        for x in s0_ids:
            for seed_key in sorted(prov_map.get(x, set())):
                seed_wid = next((ss["openalex_id"] for ss in seeds if ss["seed_key"] == seed_key), "")
                if not seed_wid:
                    continue
                w_edge.writerow({
                    "src_openalex_id": x,
                    "dst_openalex_id": seed_wid,
                    "edge_type": "cites",
                    "hop_generated": 0,
                    "from_seed": seed_key,
                })

    # works.csv: ALL S0 works (eligible + non-eligible + missing)
    processed = 0
    missing_count = 0
    tier_counts: Dict[str, int] = {"HIGH": 0, "MID": 0, "LOW": 0, "EXCLUDE": 0, "MISSING": 0}
    eligible_count = 0

    with open(works_path, "w", newline="", encoding="utf-8") as f_work:
        w_work = csv.DictWriter(f_work, fieldnames=WORKS_HEADER)
        w_work.writeheader()

        for wid in s0_ids:
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(f"[INFO] processed {processed}/{len(s0_ids)} ...", file=sys.stderr)

            try:
                raw = client.fetch_work(wid)
            except Exception as e:
                # If fetch fails, still output a minimal row and mark as missing
                tier_counts["MISSING"] += 1
                missing_count += 1

                seeds_for_x = prov_map.get(wid, set())
                route_sources = "cited_by_" + ("_".join(sorted(seeds_for_x)) if seeds_for_x else "UNKNOWN")

                axis_temporal, axis_target, cls, conf, evidence = ("uncertain", "uncertain", "UNK", "LOW", "")
                w_work.writerow({
                    "openalex_id": wid,
                    "doi": "",
                    "title": "",
                    "publication_year": "",
                    "type": "",
                    "venue": "",
                    "language": "",
                    "authors": "",
                    "institutions": "",
                    "cited_by_count": "",
                    "referenced_works_count": "",
                    "referenced_works": "",
                    "cited_by_api_url": "",
                    "has_abstract": 0,
                    "abstract": "",
                    "concepts_top": "",
                    "topics_top": "",
                    "route_sources": route_sources,
                    "first_seen_hop": 0,
                    "first_seen_from": "HLP_SEED_UNION",
                    "metadata_status": "MISSING_SEVERE",
                    "metadata_reason": "fetch_failed",
                    "relevance_score": "",
                    "relevance_tier": "MISSING",
                    "relevance_reason": f"fetch_failed:{safe_str(e)[:120]}",
                    "is_eligible": 0,
                    "axis_temporal": axis_temporal,
                    "axis_target": axis_target,
                    "class_2x2": cls,
                    "class_confidence": conf,
                    "class_evidence": evidence,
                })
                continue

            parsed = parse_work_json(raw)

            # provenance tag
            seeds_for_x = prov_map.get(wid, set())
            if len(seeds_for_x) == 0:
                route_sources = "cited_by_UNKNOWN"
            elif len(seeds_for_x) == 1:
                route_sources = f"cited_by_{next(iter(seeds_for_x))}"
            else:
                route_sources = "cited_by_BOTH"

            # Block IV: missing intercept BEFORE gate
            severe_missing, metadata_status, metadata_reason = severe_missing_check(parsed)

            # Block III: 2x2 classification (always produced; missing => uncertain/UNK)
            axis_temporal, axis_target, cls, conf, evidence = classify_2x2(
                parsed["title"], parsed["abstract"], parsed["concepts_top"], parsed["topics_top"]
            )

            # Block II: gate (skip if severe missing)
            if severe_missing:
                relevance_score = ""
                relevance_tier = "MISSING"
                relevance_reason = "missing:title+abstract+concepts+topics"
                is_eligible = 0
                tier_counts["MISSING"] += 1
                missing_count += 1
            else:
                score, tier, reason = gate.score_and_tier(
                    parsed["title"], parsed["abstract"], parsed["concepts_top"], parsed["topics_top"]
                )
                relevance_score = f"{score:.3f}"
                relevance_tier = tier
                relevance_reason = reason
                is_eligible = 1 if tier in ("HIGH", "MID") else 0
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                eligible_count += is_eligible

            w_work.writerow({
                "openalex_id": parsed["openalex_id"],
                "doi": parsed["doi"],
                "title": parsed["title"],
                "publication_year": parsed["publication_year"],
                "type": parsed["type"],
                "venue": parsed["venue"],
                "language": parsed["language"],
                "authors": parsed["authors"],
                "institutions": parsed["institutions"],
                "cited_by_count": parsed["cited_by_count"],
                "referenced_works_count": parsed["referenced_works_count"],
                "referenced_works": parsed["referenced_works"],
                "cited_by_api_url": parsed["cited_by_api_url"],
                "has_abstract": parsed["has_abstract"],
                "abstract": parsed["abstract"],
                "concepts_top": parsed["concepts_top"],
                "topics_top": parsed["topics_top"],
                "route_sources": route_sources,
                "first_seen_hop": 0,
                "first_seen_from": "HLP_SEED_UNION",
                "metadata_status": metadata_status,
                "metadata_reason": metadata_reason,
                "relevance_score": relevance_score,
                "relevance_tier": relevance_tier,
                "relevance_reason": relevance_reason,
                "is_eligible": is_eligible,
                "axis_temporal": axis_temporal,
                "axis_target": axis_target,
                "class_2x2": cls,
                "class_confidence": conf,
                "class_evidence": evidence,
            })

    runlog = {
        "run_started_utc": now_utc_iso(),
        "args": vars(args),
        "api_key_source": "cli" if args.api_key else ("env:OPENALEX_API_KEY" if os.environ.get("OPENALEX_API_KEY") else "none"),
        "gate_config": asdict(gate_cfg),
        "seeds": seed_meta,
        "s0_size": len(s0_ids),
        "counts": {
            "eligible": eligible_count,
            "missing_severe": missing_count,
            "tiers": tier_counts,
        },
        "outputs": {
            "works_csv": works_path,
            "edges_csv": edges_path,
            "runlog_json": runlog_path,
        },
    }
    with open(runlog_path, "w", encoding="utf-8") as f:
        json.dump(runlog, f, ensure_ascii=False, indent=2)

    print(f"[DONE] outputs written to: {out_dir}")
    print(f"  - works.csv: {works_path}")
    print(f"  - edges.csv: {edges_path}")
    print(f"  - runlog.json: {runlog_path}")
    print(f"[SUMMARY] S0={len(s0_ids)} | eligible={eligible_count} | missing_severe={missing_count} | tiers={tier_counts}")


# -------------------------
# CLI
# -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Route A (S0-only): seeds -> S0 -> missing -> gate -> 2x2 -> outputs")
    p.add_argument(
        "--seed",
        action="append",
        required=True,
        help="Seed in format SEEDKEY=DOI. Example: --seed HLP_JOF=10.1111/j.1540-6261.1994.tb00081.x",
    )
    p.add_argument("--out-dir", default="out_routeA_s0", help="Output directory (works.csv, edges.csv, runlog.json)")
    p.add_argument("--mailto", default=None, help="Optional mailto for OpenAlex polite pool")
    p.add_argument("--base-url", default="https://api.openalex.org", help="OpenAlex API base url")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries on transient errors")
    p.add_argument("--backoff-base", type=float, default=0.6, help="Backoff base seconds")
    p.add_argument("--backoff-jitter", type=float, default=0.2, help="Backoff jitter seconds")
    p.add_argument("--user-agent", default="routeA_s0_only/1.1", help="Custom User-Agent")

    # API key support (env var OPENALEX_API_KEY is also supported)
    p.add_argument("--api-key", default=None, help="API key/token (overrides env OPENALEX_API_KEY)")
    p.add_argument("--api-key-header", default="Authorization", help="Header name for API key (default Authorization)")
    p.add_argument(
        "--api-key-bearer",
        dest="api_key_bearer",
        action="store_true",
        help="Wrap api-key as 'Bearer <key>' unless it already starts with 'Bearer ' (default ON)",
    )
    p.add_argument(
        "--no-api-key-bearer",
        dest="api_key_bearer",
        action="store_false",
        help="Do not wrap api-key with Bearer (use raw key value). Useful with X-API-KEY headers.",
    )
    p.set_defaults(api_key_bearer=True)

    # pagination
    p.add_argument("--per-page", type=int, default=200, help="OpenAlex per-page (max 200)")
    p.add_argument("--forward-cap-seed", type=int, default=200000, help="Cap on cited-by retrieval per seed")
    p.add_argument("--verbose-fetch", action="store_true", help="Verbose fetching progress")

    # Gate params (A: CLI adjustable; bonus default 0)
    p.add_argument("--w-title", type=float, default=4.0)
    p.add_argument("--w-abs", type=float, default=3.0)
    p.add_argument("--w-concept", type=float, default=1.0)
    p.add_argument("--bonus-iv-surface", type=float, default=0.0, help="IV surface bonus (default 0; kept but off)")
    p.add_argument("--penalty-exclude", type=float, default=4.0)
    p.add_argument("--threshold-high", type=float, default=10.0)
    p.add_argument("--threshold-mid", type=float, default=6.0)

    p.add_argument("--progress-every", type=int, default=100, help="Print progress every N works (0 disables)")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
