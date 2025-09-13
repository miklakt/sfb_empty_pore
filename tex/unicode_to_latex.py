#!/usr/bin/env python3
"""
bibtex_unicode2latex.py - Replace non-ASCII characters in .bib with LaTeX macros

Usage:
    python bibtex_unicode2latex.py myrefs.bib
    # (will modify file in place)

Dependencies:
    pip install unidecode pylatexenc
"""

import sys
from pylatexenc.latexencode import unicode_to_latex

def fix_bibfile(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Convert Unicode → LaTeX
    converted = unicode_to_latex(text, non_ascii_only=True)

    if converted != text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(converted)
        print(f"[OK] Updated file in place: {path}")
    else:
        print(f"[OK] No changes needed in {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bibtex_unicode2latex.py file.bib")
        sys.exit(1)
    fix_bibfile(sys.argv[1])
