"""Harvest candidate references from Crossref for the SAINTEL literature review.

Why Crossref and not a web search: Crossref returns *authoritative* bibliographic
metadata -- DOI, journal title, ISSN, year, authors -- which is what makes a
citation verifiable. Search-engine snippets do not, and citing from a snippet is
how fabricated references enter a bibliography.

Two filters do the heavy lifting:
  type=journal-article   excludes conference proceedings and book chapters.
                         SCIE indexes journals, so proceedings cannot qualify.
  ISSN capture           SCIE status is a property of the JOURNAL, so the ISSN
                         is what you check against Clarivate's Master Journal
                         List. This script cannot check that -- it records the
                         ISSN so you can.

Nothing here asserts SCIE membership. It produces verifiable candidates.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "litreview"
MAILTO = "research@example.org"   # Crossref polite-pool identifier

THEMES: dict[str, str] = {
    "indic_ner":        "named entity recognition Indic languages Tamil low-resource",
    "codemix_ner":      "code-mixed code-switching named entity recognition social media",
    "tokenization":     "subword tokenization morphologically rich languages segmentation impact",
    "gazetteer":        "gazetteer lexicon augmented named entity recognition external knowledge",
    "cti_nlp":          "cyber threat intelligence extraction natural language processing",
    "darkweb":          "dark web threat intelligence monitoring underground forums",
    "telegram_osint":   "Telegram open source intelligence cybercrime channels analysis",
    "cni_threat":       "critical infrastructure cyber threat detection intelligence",
    "threat_graph":     "threat intelligence knowledge graph STIX ontology sharing",
    "crosslingual":     "cross-lingual transfer multilingual transformer low-resource languages",
    "unicode_norm":     "text normalization Unicode Indic script processing",
    "annotation":       "corpus annotation inter-annotator agreement named entity guidelines",
    "social_threat":    "social media threat detection classification security OSINT",
    "multiling_cti":    "multilingual non-English cyber threat intelligence analysis",
}


def query(term: str, rows: int = 40) -> list[dict]:
    # `requests` rather than urllib: the macOS python.org build ships without
    # root certificates, so urllib raises CERTIFICATE_VERIFY_FAILED here.
    response = requests.get(
        "https://api.crossref.org/works",
        params={
            "query.bibliographic": term,
            "filter": "type:journal-article,from-pub-date:2018-01-01",
            "rows": rows,
            "select": "DOI,title,author,container-title,ISSN,issued,is-referenced-by-count,URL",
            "sort": "relevance",
            "mailto": MAILTO,
        },
        headers={"User-Agent": f"SAINTEL-litreview (mailto:{MAILTO})"},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["message"]["items"]


def tidy(item: dict, theme: str) -> dict | None:
    title = (item.get("title") or [""])[0].strip()
    journal = (item.get("container-title") or [""])[0].strip()
    if not title or not journal:
        return None
    authors = item.get("author") or []
    names = [f"{a.get('family','')}, {a.get('given','')}".strip(", ") for a in authors[:6]]
    year = None
    for key in ("issued",):
        parts = (item.get(key) or {}).get("date-parts") or [[None]]
        year = parts[0][0]
    return {
        "theme": theme,
        "title": title,
        "authors": names,
        "journal": journal,
        "issn": item.get("ISSN") or [],
        "year": year,
        "doi": item.get("DOI"),
        "cited_by": item.get("is-referenced-by-count", 0),
        "url": item.get("URL"),
    }


def main() -> int:
    seen: set[str] = set()
    records: list[dict] = []
    for theme, term in THEMES.items():
        try:
            items = query(term)
        except Exception as exc:
            print(f"  {theme:16s} FAILED {type(exc).__name__}", file=sys.stderr)
            continue
        kept = 0
        for item in items:
            rec = tidy(item, theme)
            if rec is None or rec["doi"] in seen:
                continue
            seen.add(rec["doi"])
            records.append(rec)
            kept += 1
        print(f"  {theme:16s} {kept:3d} journal articles")
        time.sleep(1)   # be polite to the API

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(records)} unique journal articles -> {OUT/'candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
