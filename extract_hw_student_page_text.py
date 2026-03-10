#!/usr/bin/env python3
"""Fetch shortlisted Heriot-Watt links and extract main page text.

Usage:
  python extract_hw_student_page_text.py
  python extract_hw_student_page_text.py --input hw_student_links_clean.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "hw_student_links_clean.json"
DEFAULT_OUTPUT_JSON = ROOT / "hw_student_pages_extracted.json"
DEFAULT_OUTPUT_TEXT = ROOT / "agentic_rag" / "knowledge_extracted_pages.txt"
DEFAULT_LOG = ROOT / "hw_text_extraction.log"
DEFAULT_USER_AGENT = (
    "UniBotPageExtractor/1.0 (+https://example.invalid/contact; purpose=academic-research)"
)
NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "nav",
    "footer",
    "header",
    ".navigation",
    ".breadcrumbs",
    ".breadcrumb",
    ".cookie",
    ".consent",
    ".sidebar",
    ".share",
    ".social-share",
    ".menu",
    ".newsletter",
    ".promo",
    ".hero",
    ".related-content",
)
BAD_LINE_PATTERNS = (
    r"^\s*search\s*$",
    r"^\s*menu\s*$",
    r"^\s*skip to",
    r"^\s*back to top\s*$",
    r"^\s*share this",
    r"^\s*follow us",
    r"^\s*cookie",
    r"^\s*privacy",
)


@dataclass(slots=True)
class ExtractedRecord:
    category: str
    site_key: str
    site_label: str
    title: str
    url: str
    source_url: str
    summary: str
    headings: list[str]
    extracted_text: str
    extracted_at: str


def configure_logging(log_file: Path) -> None:
    logging.basicConfig(
        filename=str(log_file),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(console)


def make_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_noise(container: Tag) -> None:
    for selector in NOISE_SELECTORS:
        for node in container.select(selector):
            node.decompose()


def best_content_container(soup: BeautifulSoup) -> Tag | None:
    preferred_selectors = [
        "main",
        "article",
        "[role='main']",
        ".post-content",
        ".entry-content",
        ".content",
        ".page-content",
        ".content-area",
        ".main-content",
        ".detail",
    ]
    for selector in preferred_selectors:
        node = soup.select_one(selector)
        if node and len(" ".join(node.stripped_strings)) > 200:
            return node
    candidates = [
        node
        for node in soup.find_all(["section", "div"])
        if len(node.get_text(" ", strip=True)) > 200
    ]
    if not candidates:
        return soup.body
    return max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))


def is_bad_line(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    for pattern in BAD_LINE_PATTERNS:
        if re.search(pattern, lowered):
            return True
    if len(lowered) == 1:
        return True
    return False


def dedupe_lines(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        cleaned = normalize_whitespace(line)
        if not cleaned or is_bad_line(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def extract_text_from_container(container: Tag) -> str:
    blocks: list[str] = []
    for node in container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = normalize_whitespace(node.get_text(" ", strip=True))
        if not text or is_bad_line(text):
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            blocks.append(f"\n{text}\n")
        elif node.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)
    deduped = dedupe_lines(blocks)
    return normalize_whitespace("\n".join(deduped))


def fetch_html(session: requests.Session, url: str, timeout: float) -> str | None:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        if "html" not in response.headers.get("content-type", "").lower():
            logging.warning("Skipping non-HTML content: %s", url)
            return None
        return response.text
    except requests.RequestException as exc:
        logging.warning("Failed to fetch %s: %s", url, exc)
        return None


def extract_page_text(html: str) -> tuple[str, list[str], str]:
    soup = BeautifulSoup(html, "html.parser")
    container = best_content_container(soup)
    if container is None:
        return "", [], ""
    remove_noise(container)
    text = extract_text_from_container(container)
    headings = []
    for heading in container.find_all(["h1", "h2", "h3"])[:8]:
        value = normalize_whitespace(heading.get_text(" ", strip=True))
        if value and value.lower() not in {h.lower() for h in headings}:
            headings.append(value)
    title = ""
    if soup.title:
        title = normalize_whitespace(soup.title.get_text(" ", strip=True))
    return text, headings, title


def load_links(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("links", []))


def build_knowledge_text(records: Sequence[ExtractedRecord]) -> str:
    category_titles = {
        "academics": "Academics",
        "student_support": "Student Support",
        "accommodation_and_living": "Accommodation and Living",
        "student_life": "Student Life",
        "sports_and_activity": "Sports and Activity",
        "campus_and_services": "Campus and Services",
        "fees_and_funding": "Fees and Funding",
    }
    lines: list[str] = []
    lines.append("# Heriot-Watt Student Knowledge Base")
    lines.append("")
    lines.append("This file contains extracted page text from curated student-facing Heriot-Watt links.")
    lines.append("")
    lines.append(f"Total extracted pages: {len(records)}")
    lines.append("")

    current_category = None
    for record in records:
        if record.category != current_category:
            current_category = record.category
            lines.append(f"## {category_titles.get(record.category, record.category.title())}")
            lines.append("")
        lines.append(f"Title: {record.title}")
        lines.append(f"Category: {category_titles.get(record.category, record.category.title())}")
        lines.append(f"Site: {record.site_label}")
        lines.append(f"URL: {record.url}")
        if record.summary:
            lines.append(f"Summary: {record.summary}")
        if record.headings:
            lines.append(f"Headings: {' | '.join(record.headings)}")
        lines.append("Content:")
        lines.append(record.extracted_text)
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch shortlisted Heriot-Watt links and extract main page text."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-text", default=str(DEFAULT_OUTPUT_TEXT))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_json = Path(args.output_json)
    output_text = Path(args.output_text)
    log_file = Path(args.log_file)
    timeout = max(1.0, float(args.timeout))
    delay = max(0.0, float(args.delay))

    configure_logging(log_file)
    links = load_links(input_path)
    session = make_session(args.user_agent)

    extracted: list[ExtractedRecord] = []
    for index, item in enumerate(links, start=1):
        url = str(item["url"])
        logging.info("Extracting %s/%s %s", index, len(links), url)
        html = fetch_html(session, url, timeout)
        if html is None:
            continue
        text, headings, page_title = extract_page_text(html)
        if len(text) < 120:
            logging.warning("Low-text page skipped: %s", url)
            continue
        extracted.append(
            ExtractedRecord(
                category=str(item["category"]),
                site_key=str(item["site_key"]),
                site_label=str(item["site_label"]),
                title=page_title or str(item.get("title") or item.get("anchor_text") or url),
                url=url,
                source_url=str(item.get("source_url") or url),
                summary=str(item.get("description") or "").strip(),
                headings=headings,
                extracted_text=text,
                extracted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        )
        if delay:
            time.sleep(delay)

    extracted.sort(key=lambda item: (item.category, item.site_key, item.title.lower(), item.url))
    output_json.write_text(
        json.dumps(
            {
                "total_pages": len(extracted),
                "pages": [asdict(record) for record in extracted],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_text.write_text(build_knowledge_text(extracted), encoding="utf-8")

    logging.info("Wrote %s extracted pages", len(extracted))
    print(f"Wrote {len(extracted)} extracted pages to {output_json} and {output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
