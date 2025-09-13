#!/usr/bin/env python3
"""
biblio_authors_fix.py — Normalize and update BibTeX author fields using web metadata.

Usage:
    python biblio_authors_fix.py biblio.bib -o biblio_authors_fixed.bib --email you@example.com

Dependencies:
    pip install bibtexparser requests tqdm ratelimit backoff
"""

import argparse, os, re, sys, html
import requests
import backoff
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter

CROSSREF = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1"

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def norm_author_list(authors):
    """
    Normalize author list for biblatex:
    - Format: 'Last, First and Last, First'
    - Handles unicode safely
    """
    clean = []
    for a in authors:
        first = " ".join(a.get("given", "").split())
        last = " ".join(a.get("family", "").split())
        if last and first:
            clean.append(f"{last}, {first}")
        elif last:
            clean.append(last)
    return " and ".join(clean)

def extract_doi(entry):
    return (entry.get("doi") or "").strip().lower()

def extract_title(entry):
    if "title" not in entry: return None
    return re.sub(r"[{}]", "", entry["title"]).strip()

# -------------------------------------------------------------------------
# HTTP utilities with polite rate limits
# -------------------------------------------------------------------------

@sleep_and_retry
@limits(calls=3, period=1)
def _get(session, url, **kw):
    return session.get(url, timeout=20, **kw)

@backoff.on_exception(backoff.expo, requests.RequestException, max_time=60)
def crossref_lookup(session, doi=None, title=None):
    if doi:
        r = _get(session, f"{CROSSREF}/{requests.utils.quote(doi)}")
        if r.status_code == 200:
            return r.json().get("message", {})
    if title:
        r = _get(session, CROSSREF, params={"query.title": title, "rows": 1})
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                return items[0]
    return {}

@backoff.on_exception(backoff.expo, requests.RequestException, max_time=60)
def semanticscholar_lookup(session, doi=None, title=None):
    fields = "title,authors,externalIds"
    if doi:
        r = _get(session, f"{SEMANTIC_SCHOLAR}/paper/DOI:{doi}", params={"fields": fields})
        if r.status_code == 200: return r.json()
    if title:
        r = _get(session, f"{SEMANTIC_SCHOLAR}/paper/search", params={"query": title, "fields": fields, "limit":1})
        if r.status_code == 200 and r.json().get("data"):
            return r.json()["data"][0]
    return {}

# -------------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------------

def fix_authors(entry, session, overwrite=False):
    """Update author field of a bib entry using Crossref / Semantic Scholar"""
    doi = extract_doi(entry)
    title = extract_title(entry)
    authors_new = None

    # Try Crossref
    data = crossref_lookup(session, doi=doi or None, title=(None if doi else title))
    if data and "author" in data:
        authors_new = norm_author_list(data["author"])

    # Fallback Semantic Scholar
    if not authors_new:
        data = semanticscholar_lookup(session, doi=doi or None, title=(None if doi else title))
        if data and "authors" in data:
            auths = [{"given": a.get("name").split()[0], "family": " ".join(a.get("name").split()[1:])} for a in data["authors"]]
            authors_new = norm_author_list(auths)

    if authors_new and (overwrite or not entry.get("author")):
        entry["author"] = authors_new
        return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Normalize and update BibTeX author fields.")
    ap.add_argument("bibfile", help="Input .bib file")
    ap.add_argument("-o", "--output", help="Output .bib file")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing author fields")
    ap.add_argument("--email", help="Email for polite User-Agent (recommended)")
    args = ap.parse_args()

    parser = BibTexParser(common_strings=True)
    with open(args.bibfile, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f, parser=parser)

    session = requests.Session()
    ua = "biblio-authors-fix/1.0"
    if args.email: ua += f" (mailto:{args.email})"
    session.headers.update({"User-Agent": ua})

    fixed = 0
    for entry in tqdm(db.entries, desc="Fixing authors", unit="entry"):
        if fix_authors(entry, session, overwrite=args.overwrite):
            fixed += 1

    writer = BibTexWriter()
    writer.order_entries_by = None
    out = args.output or os.path.splitext(args.bibfile)[0] + ".authors_fixed.bib"
    with open(out, "w", encoding="utf-8") as f:
        f.write(writer.write(db))

    print(f"Done. Updated authors in {fixed} entries. Wrote {out}")

if __name__ == "__main__":
    main()
