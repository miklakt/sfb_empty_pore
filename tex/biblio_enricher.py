#!/usr/bin/env python3

"""
biblio_enricher.py — fill missing DOI and abstract fields in a BibTeX file.

Sources (in this order):
  1) Crossref (by DOI if present; otherwise by title+author+year)
  2) Semantic Scholar (title search or DOI lookup)
  3) arXiv (by arXiv ID if present; otherwise title search)
  4) PubMed (title+year search)

Usage:
  python biblio_enricher.py biblio.bib -o biblio_enriched.bib --email you@example.com
  python biblio_enricher.py biblio.bib --dry-run

Install deps:
  pip install bibtexparser requests tqdm backoff ratelimit

Notes:
- Providing --email sets a polite User-Agent for Crossref.
- By default we DO NOT overwrite existing 'doi' or 'abstract' unless --overwrite is set.
- A local JSON cache speeds up repeated runs and reduces API load.
- Abstracts are normalized to single-line, LaTeX-safe text.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import requests
from ratelimit import limits, sleep_and_retry
import backoff
from tqdm import tqdm

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.customization import homogenize_latex_encoding
except Exception as e:
    print("Missing dependency bibtexparser. Install with: pip install bibtexparser", file=sys.stderr)
    raise

CACHE_FILE = ".biblio_enricher_cache.json"
CROSSREF_BASE = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "http://export.arxiv.org/api/query"
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# ------------------------------- Utilities ----------------------------------

def load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(path: str, cache: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def cache_key(kind: str, payload: Dict[str, Any]) -> str:
    m = hashlib.sha256()
    m.update(kind.encode())
    m.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return m.hexdigest()

def first_or_none(seq):
    return seq[0] if seq else None

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def latex_escape(text: str) -> str:
    # Basic LaTeX escaping + keep braces which bibtex uses
    replacements = {
        "\\": r"\\",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    # Remove nulls and control chars
    s = "".join(out)
    s = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", s)
    return norm_space(s)

def extract_year(entry: dict) -> Optional[int]:
    for key in ("year", "date", "pubyear"):
        v = entry.get(key)
        if v:
            m = re.search(r"\d{4}", v)
            if m:
                return int(m.group(0))
    return None

def extract_title(entry: dict) -> Optional[str]:
    for key in ("title", "booktitle"):
        v = entry.get(key)
        if v:
            # Strip outer braces and LaTeX commands for searching
            t = re.sub(r"{|}", "", v)
            t = re.sub(r"\\[a-zA-Z]+\s*\{[^}]*\}", "", t)
            t = re.sub(r"\\[a-zA-Z]+", "", t)
            return norm_space(t)
    return None

def extract_first_author_surname(entry: dict) -> Optional[str]:
    authors = entry.get("author")
    if not authors:
        return None
    # Split authors by ' and ' in BibTeX format
    first = authors.split(" and ")[0]
    # BibTeX names can be "Last, First" or "First Last"
    if "," in first:
        last = first.split(",")[0]
    else:
        last = first.split()[-1]
    return norm_space(re.sub(r"[{}]", "", last))

# ------------------------------- HTTP + Backoff ------------------------------

def make_session(email: Optional[str]) -> requests.Session:
    s = requests.Session()
    ua = f"biblio-enricher/1.0 (+https://example.org)"
    if email:
        ua += f" mailto:{email}"
    s.headers.update({"User-Agent": ua})
    return s

# Respect polite rate limits (Crossref guideline ~50 rps max; we'll be conservative)
@sleep_and_retry
@limits(calls=3, period=1)  # 3 req/s per host default
def _get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    return session.get(url, timeout=20, **kwargs)

@sleep_and_retry
@limits(calls=3, period=1)
def _post(session: requests.Session, url: str, **kwargs) -> requests.Response:
    return session.post(url, timeout=20, **kwargs)

def backoff_hdlr(details):
    print(f"Backing off {details['wait']:0.1f}s after {details['tries']} tries for {details['target'].__name__}...", file=sys.stderr)

# ------------------------------- Crossref ------------------------------------

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60, on_backoff=backoff_hdlr)
def crossref_lookup_by_doi(session: requests.Session, doi: str, cache: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    doi = doi.lower().strip()
    key = cache_key("crossref_doi", {"doi": doi})
    if key in cache: return cache[key]
    r = _get(session, f"{CROSSREF_BASE}/{requests.utils.quote(doi)}")
    if r.status_code != 200:
        cache[key] = (None, None); return (None, None)
    data = r.json().get("message", {})
    abstract = data.get("abstract")
    if abstract:
        # Crossref abstracts can be JATS XML (<jats:p> ...)
        abstract = re.sub(r"<\/?jats:[^>]+>", " ", abstract)
        abstract = re.sub(r"<[^>]+>", " ", abstract)
        abstract = html.unescape(abstract)
    result = (data.get("DOI"), abstract)
    cache[key] = result
    return result

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60, on_backoff=backoff_hdlr)
def crossref_search(session: requests.Session, title: str, author_last: Optional[str], year: Optional[int], cache: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    params = {"query.title": title, "rows": 3}
    if author_last:
        params["query.author"] = author_last
    if year:
        params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
    key = cache_key("crossref_search", {"title": title, "author": author_last, "year": year})
    if key in cache: return cache[key]
    r = _get(session, CROSSREF_BASE, params=params)
    if r.status_code != 200:
        cache[key] = (None, None); return (None, None)
    items = r.json().get("message", {}).get("items", [])
    if not items:
        cache[key] = (None, None); return (None, None)
    # Pick best: exact-ish title match if possible
    def score(item):
        tlist = item.get("title") or []
        t = norm_space(" ".join(tlist)).lower()
        s = 0
        if title.lower() in t or t in title.lower():
            s += 2
        if year:
            item_year = first_or_none(item.get("issued", {}).get("date-parts", [[None]])[0])
            if item_year == year: s += 1
        if author_last:
            auths = " ".join([a.get("family","") for a in item.get("author", [])]).lower()
            if author_last.lower() in auths: s += 1
        return s
    best = sorted(items, key=score, reverse=True)[0]
    doi = best.get("DOI")
    abstract = best.get("abstract")
    if abstract:
        abstract = re.sub(r"<\/?jats:[^>]+>", " ", abstract)
        abstract = re.sub(r"<[^>]+>", " ", abstract)
        abstract = html.unescape(abstract)
    result = (doi, abstract)
    cache[key] = result
    return result

# --------------------------- Semantic Scholar --------------------------------

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60, on_backoff=backoff_hdlr)
def s2_lookup(session: requests.Session, title: Optional[str]=None, doi: Optional[str]=None, cache: Dict[str, Any]={}) -> Tuple[Optional[str], Optional[str]]:
    fields = "title,abstract,externalIds"
    if doi:
        key = cache_key("s2_doi", {"doi": doi})
        if key in cache: return cache[key]
        r = _get(session, f"{SEMANTIC_SCHOLAR_BASE}/paper/DOI:{requests.utils.quote(doi)}", params={"fields": fields})
        if r.status_code != 200:
            cache[key] = (None, None); return (None, None)
        data = r.json()
        doi_out = (data.get("externalIds") or {}).get("DOI") or doi
        abstract = data.get("abstract")
        res = (doi_out, abstract)
        cache[key] = res
        return res
    if title:
        key = cache_key("s2_title", {"title": title})
        if key in cache: return cache[key]
        r = _get(session, f"{SEMANTIC_SCHOLAR_BASE}/paper/search", params={"query": title, "fields": fields, "limit": 3})
        if r.status_code != 200:
            cache[key] = (None, None); return (None, None)
        data = r.json()
        if not data.get("data"):
            cache[key] = (None, None); return (None, None)
        best = data["data"][0]
        doi_out = (best.get("externalIds") or {}).get("DOI")
        abstract = best.get("abstract")
        res = (doi_out, abstract)
        cache[key] = res
        return res
    return (None, None)

# --------------------------------- arXiv -------------------------------------

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60, on_backoff=backoff_hdlr)
def arxiv_lookup(session: requests.Session, arxiv_id: Optional[str]=None, title: Optional[str]=None, cache: Dict[str, Any]={}) -> Tuple[Optional[str], Optional[str]]:
    if arxiv_id:
        key = cache_key("arxiv_id", {"id": arxiv_id})
        if key in cache: return cache[key]
        q = f"id_list={arxiv_id}"
    elif title:
        key = cache_key("arxiv_title", {"title": title})
        if key in cache: return cache[key]
        q = "search_query=ti:" + requests.utils.quote(f'"{title}"')
    else:
        return (None, None)
    r = _get(session, f"{ARXIV_BASE}?{q}&max_results=1")
    if r.status_code != 200:
        if arxiv_id:
            cache[key] = (None, None)
        return (None, None)
    # Very light Atom parsing
    text = r.text
    # DOI
    mdoi = re.search(r"<arxiv:doi>([^<]+)</arxiv:doi>", text)
    doi = mdoi.group(1).strip() if mdoi else None
    # Abstract
    mabs = re.search(r"<summary>(.*?)</summary>", text, flags=re.DOTALL)
    abstract = None
    if mabs:
        abstract = html.unescape(mabs.group(1))
    if arxiv_id:
        cache[key] = (doi, abstract)
    return (doi, abstract)

# -------------------------------- PubMed -------------------------------------

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60, on_backoff=backoff_hdlr)
def pubmed_lookup(session: requests.Session, title: str, year: Optional[int], cache: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    key = cache_key("pubmed", {"title": title, "year": year})
    if key in cache: return cache[key]
    params = {"db":"pubmed", "retmode":"json", "term": title, "retmax": 3}
    if year:
        params["mindate"] = year
        params["maxdate"] = year
    r = _get(session, PUBMED_SEARCH, params=params)
    if r.status_code != 200:
        cache[key] = (None, None); return (None, None)
    ids = (r.json().get("esearchresult", {}) or {}).get("idlist", [])
    if not ids:
        cache[key] = (None, None); return (None, None)
    pmid = ids[0]
    r2 = _get(session, PUBMED_FETCH, params={"db":"pubmed", "retmode":"xml", "id": pmid})
    if r2.status_code != 200:
        cache[key] = (None, None); return (None, None)
    xml = r2.text
    mdoi = re.search(r"<ArticleId IdType=\"doi\">([^<]+)</ArticleId>", xml)
    doi = mdoi.group(1).strip() if mdoi else None
    mabs = re.search(r"<Abstract>(.*?)</Abstract>", xml, flags=re.DOTALL)
    abstract = None
    if mabs:
        # concatenate AbstractText blocks
        parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", mabs.group(1), flags=re.DOTALL)
        abstract = norm_space(html.unescape(" ".join(parts)))
    res = (doi, abstract)
    cache[key] = res
    return res

# ---------------------------- Core enrichment --------------------------------

@dataclass
class EnrichResult:
    updated: bool
    added_doi: bool
    added_abstract: bool
    doi: Optional[str]
    abstract: Optional[str]
    source: Optional[str]

def enrich_entry(entry: dict, session: requests.Session, cache: Dict[str, Any], overwrite: bool=False) -> EnrichResult:
    # Extract hints
    title = extract_title(entry)
    year = extract_year(entry)
    last = extract_first_author_surname(entry)
    existing_doi = (entry.get("doi") or "").strip()
    arxiv_id = None
    for k in ("eprint", "arxivid", "archiveprefix", "howpublished", "url"):
        v = entry.get(k, "")
        m = re.search(r"arxiv\.org\/abs\/([0-9]+\.[0-9]+|[a-z\-]+\/[0-9]+)", v, flags=re.I)
        if m:
            arxiv_id = m.group(1)
            break

    doi, abstract, source = None, None, None

    # 1) If DOI exists, use it to fetch metadata quickly
    if existing_doi:
        d, a = crossref_lookup_by_doi(session, existing_doi, cache)
        if not a:
            # Semantic Scholar sometimes has abstracts when Crossref doesn't
            d2, a2 = s2_lookup(session, doi=existing_doi, cache=cache)
            d = d or d2
            a = a or a2
        if d: doi = d; source = source or "crossref/doi"
        if a: abstract = a; source = source or "crossref/semanticscholar"
    # 2) If still missing DOI/abstract, try Crossref by title
    if (not doi or not abstract) and title:
        d, a = crossref_search(session, title, last, year, cache)
        if d and not doi: doi = d; source = source or "crossref/search"
        if a and not abstract: abstract = a; source = source or "crossref/search"
    # 3) Semantic Scholar title search (often has abstracts)
    if (not doi or not abstract) and title:
        d, a = s2_lookup(session, title=title, cache=cache)
        if d and not doi: doi = d; source = source or "semanticscholar"
        if a and not abstract: abstract = a; source = source or "semanticscholar"
    # 4) arXiv
    if (not doi or not abstract) and (arxiv_id or title):
        d, a = arxiv_lookup(session, arxiv_id=arxiv_id, title=(None if arxiv_id else title), cache=cache)
        if d and not doi: doi = d; source = source or "arxiv"
        if a and not abstract: abstract = a; source = source or "arxiv"
    # 5) PubMed (biomed)
    if (not doi or not abstract) and title:
        d, a = pubmed_lookup(session, title, year, cache)
        if d and not doi: doi = d; source = source or "pubmed"
        if a and not abstract: abstract = a; source = source or "pubmed"

    # Normalize abstract
    if abstract:
        abstract = latex_escape(norm_space(abstract))

    updated = False
    added_doi = False
    added_abs = False

    if doi and (overwrite or not entry.get("doi")):
        entry["doi"] = doi
        updated = True
        added_doi = True
    if abstract and (overwrite or not entry.get("abstract")):
        entry["abstract"] = abstract
        updated = True
        added_abs = True

    return EnrichResult(updated, added_doi, added_abs, doi, abstract, source)

def read_bib(path: str) -> dict:
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    with open(path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f, parser=parser)
    return db

def write_bib(db, path: str) -> None:
    writer = bibtexparser.bwriter.BibTexWriter()
    writer.indent = "  "
    writer.comma_first = False
    writer.order_entries_by = ("ID",)
    with open(path, "w", encoding="utf-8") as f:
        f.write(writer.write(db))

def main():
    ap = argparse.ArgumentParser(description="Enrich BibTeX entries with DOI and abstract.")
    ap.add_argument("bibfile", help="Input .bib file")
    ap.add_argument("-o", "--output", help="Output .bib file (default: <input>.enriched.bib)")
    ap.add_argument("--email", help="Contact email for polite API User-Agent (recommended)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing fields")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without writing file")
    ap.add_argument("--cache", default=CACHE_FILE, help=f"Cache file (default: {CACHE_FILE})")
    args = ap.parse_args()

    db = read_bib(args.bibfile)
    session = make_session(args.email)
    cache = load_cache(args.cache)

    updates = 0
    added_doi = 0
    added_abs = 0

    for entry in tqdm(db.entries, desc="Enriching", unit="entry"):
        res = enrich_entry(entry, session, cache, overwrite=args.overwrite)
        if res.updated:
            updates += 1
            if res.added_doi: added_doi += 1
            if res.added_abstract: added_abs += 1

    # Save results
    save_cache(args.cache, cache)

    if args.dry_run:
        print(f"[DRY RUN] Entries to update: {updates} (doi added: {added_doi}, abstract added: {added_abs})")
        return

    out = args.output or (os.path.splitext(args.bibfile)[0] + ".enriched.bib")
    write_bib(db, out)
    print(f"Wrote {out} — updated entries: {updates} (doi added: {added_doi}, abstract added: {added_abs})")

if __name__ == "__main__":
    main()
