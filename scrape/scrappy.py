#!/usr/bin/env python3
# Scrape visible text from a local HTML file.
# Usage: python scrape_text_from_html.py path/to/file.html --out output.txt
import argparse
import re
import sys
from typing import Iterable

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'beautifulsoup4'. Install with: pip install beautifulsoup4"
    ) from exc


def _visible_texts(soup: BeautifulSoup) -> Iterable[str]:
    # Remove non-visible elements
    for tag in soup(["script", "style", "noscript", "template", "svg", "head", "title", "meta", "link"]):
        tag.decompose()
    # Get text and normalize whitespace
    text = soup.get_text(separator="\n")
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            yield line


def scrape_text_from_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return "\n".join(_visible_texts(soup))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Scrape visible text from a local HTML file")
    parser.add_argument("path", help="Path to the HTML file")
    parser.add_argument("--out", help="Output file path (optional)")
    args = parser.parse_args(argv)

    try:
        text = scrape_text_from_html(args.path)
    except OSError as exc:
        print(f"Failed to read file: {exc}", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
