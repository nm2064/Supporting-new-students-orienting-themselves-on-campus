#!/usr/bin/env python3
"""Scrape text-first society data from HW Union for RAG ingestion.

Usage:
  python scripts/scrape_hwunion_societies.py
  python scripts/scrape_hwunion_societies.py --start-url https://www.hwunion.com/get-involved/societies/
  python scripts/scrape_hwunion_societies.py --max-pages 300 --delay 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_LISTING_URL = "https://www.hwunion.com/get-involved/societies/"
DEFAULT_UA = (
    "UniBotSocietyScraper/1.0 (+https://example.invalid/contact; "
    "purpose=academic-rag-dataset)"
)
ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "scraper.log"
JSONL_FILE = ROOT / "data" / "raw" / "societies.jsonl"
TEXT_FILE = ROOT / "data" / "raw" / "societies.txt"
SOCIAL_HOST_HINTS = (
    "instagram.com",
    "facebook.com",
    "linkedin.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "tiktok.com",
    "linktr.ee",
)
BAD_SCHEMES = ("mailto:", "tel:", "javascript:", "#")


@dataclass
class SocietyRecord:
    name: str = ""
    url: str = ""
    description: str = ""
    categories: List[str] = field(default_factory=list)
    contact: Dict[str, str] = field(default_factory=lambda: {"email": "", "phone": ""})
    social_links: List[str] = field(default_factory=list)
    committee: List[str] = field(default_factory=list)
    membership_info: str = ""
    meeting_times: str = ""
    last_updated: str = ""
    raw_text: str = ""
    scraped_at: str = ""
    sources: Dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "categories": self.categories,
            "contact": self.contact,
            "social_links": self.social_links,
            "committee": self.committee,
            "membership_info": self.membership_info,
            "meeting_times": self.meeting_times,
            "last_updated": self.last_updated,
            "raw_text": self.raw_text,
            "scraped_at": self.scraped_at,
        }
        if self.sources:
            out["sources"] = self.sources
        return out


def configure_logging() -> None:
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(console)


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query_items = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False))
    query = urlencode(query_items, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def in_scope(url: str, allowed_host: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    return host == allowed_host or host.endswith("." + allowed_host)


def looks_like_society_page(url: str) -> bool:
    lower = url.lower()
    return "/societ" in lower


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


def parse_robots(start_url: str, user_agent: str, timeout: float) -> Tuple[bool, Optional[robotparser.RobotFileParser]]:
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
        if resp.status_code >= 400:
            logging.warning("robots.txt returned status %s at %s, proceeding cautiously.", resp.status_code, robots_url)
            return True, None
        rp.parse(resp.text.splitlines())
    except requests.RequestException as exc:
        logging.warning("Could not fetch robots.txt (%s), proceeding cautiously.", exc)
        return True, None
    allowed = rp.can_fetch(user_agent, start_url) and rp.can_fetch("*", start_url)
    return allowed, rp


def fetch_page(
    session: requests.Session,
    url: str,
    timeout: float,
    delay: float,
    state: Dict[str, float],
) -> Optional[str]:
    now = time.monotonic()
    wait = delay - (now - state.get("last_request_time", 0.0))
    if wait > 0:
        time.sleep(wait)
    try:
        resp = session.get(url, timeout=timeout)
        state["last_request_time"] = time.monotonic()
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("content-type", "").lower():
            logging.info("Skipping non-HTML content: %s", url)
            return None
        return resp.text
    except requests.RequestException as exc:
        logging.error("Failed request for %s: %s", url, exc)
        return None


def is_probably_js_rendered(html: str) -> bool:
    text = re.sub(r"\s+", " ", html).lower()
    js_hints = (
        "enable javascript",
        "loading...",
        "data-reactroot",
        "__next",
        "id=\"app\"",
        "id=\"root\"",
    )
    body = BeautifulSoup(html, "html.parser").body
    short_body = ""
    if body:
        short_body = " ".join(body.stripped_strings)[:300]
    return len(short_body) < 40 and any(h in text for h in js_hints)


def fetch_with_playwright(url: str, timeout_ms: int = 30000) -> Optional[str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        logging.error(
            "Playwright not installed; cannot render JS for %s. Install with "
            "`pip install playwright` and `playwright install chromium`.",
            url,
        )
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = page.content()
            browser.close()
            return html
    except Exception as exc:
        logging.error("Playwright fetch failed for %s: %s", url, exc)
        return None


def remove_noise(container: Tag) -> None:
    for selector in (
        "script",
        "style",
        "noscript",
        "svg",
        "form",
        "nav",
        "footer",
        "header",
        ".menu",
        ".navigation",
        ".breadcrumbs",
        ".breadcrumb",
        ".share",
        ".social-share",
        ".newsletter",
        ".cookie",
        ".consent",
        ".sidebar",
        ".widget",
    ):
        for node in container.select(selector):
            node.decompose()


def extract_listing_candidates(
    html: str, base_url: str, allowed_host: str
) -> Tuple[Set[str], Set[str], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    snippets: List[str] = []
    for a in anchors[:10]:
        snippets.append(str(a)[:200])

    society_links: Set[str] = set()
    pagination_links: Set[str] = set()
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href or href.startswith(BAD_SCHEMES):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if not in_scope(absolute, allowed_host):
            continue
        lower = absolute.lower()
        text = " ".join(a.stripped_strings).lower()
        if looks_like_society_page(absolute) and ("societies" in lower or "/society/" in lower or "/societies/" in lower):
            society_links.add(absolute)
            continue
        if any(x in lower for x in ("page=", "/page/", "offset=", "p=", "loadmore", "paged=")):
            pagination_links.add(absolute)
            continue
        if "next" in text or "load more" in text or "older" in text:
            pagination_links.add(absolute)
    return society_links, pagination_links, snippets


def best_content_container(soup: BeautifulSoup) -> Optional[Tag]:
    preferred_selectors = [
        "main",
        "article",
        "[role='main']",
        ".post-content",
        ".entry-content",
        ".content",
        ".society",
        ".single",
    ]
    for selector in preferred_selectors:
        node = soup.select_one(selector)
        if node and len(" ".join(node.stripped_strings)) > 120:
            return node
    candidates = [n for n in soup.find_all(["div", "section"]) if n.get_text(" ", strip=True)]
    if not candidates:
        return soup.body
    return max(candidates, key=lambda n: len(n.get_text(" ", strip=True)))


def extract_heading_and_bullets(container: Tag) -> List[str]:
    chunks: List[str] = []
    for node in container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = normalize_whitespace(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            chunks.append(f"\n{text}\n")
        elif node.name == "li":
            chunks.append(f"- {text}")
        else:
            chunks.append(text)
    return chunks


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_lines(text: str) -> str:
    seen: Set[str] = set()
    out: List[str] = []
    for line in text.splitlines():
        key = line.strip().lower()
        if not key:
            out.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return normalize_whitespace("\n".join(out))


def extract_emails_phones(text: str) -> Tuple[str, str]:
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.I)
    phone_match = re.search(r"(?:(?:\+?44|0)\s?\d[\d\s().-]{7,}\d)", text)
    return (email_match.group(0) if email_match else "", phone_match.group(0) if phone_match else "")


def extract_social_links(container: Tag, page_url: str) -> List[str]:
    links: Set[str] = set()
    for a in container.find_all("a", href=True):
        href = urljoin(page_url, a["href"].strip())
        lower = href.lower()
        if any(h in lower for h in SOCIAL_HOST_HINTS):
            links.add(normalize_url(href))
    return sorted(links)


def extract_categories(soup: BeautifulSoup, container: Tag) -> List[str]:
    categories: Set[str] = set()
    for node in soup.select(".category, .categories, .tag, .tags, [rel='tag']"):
        txt = normalize_whitespace(node.get_text(" ", strip=True))
        if 1 < len(txt) < 60:
            categories.add(txt)
    text = container.get_text("\n", strip=True)
    for line in text.splitlines():
        line_n = normalize_whitespace(line)
        if re.match(r"^(category|categories|tags?)\s*:", line_n, flags=re.I):
            value = line_n.split(":", 1)[1].strip()
            for part in re.split(r"[,/|]", value):
                p = part.strip()
                if p:
                    categories.add(p)
    return sorted(categories)


def extract_field_by_label(text: str, labels: Iterable[str]) -> str:
    lines = [normalize_whitespace(x) for x in text.splitlines() if normalize_whitespace(x)]
    for idx, line in enumerate(lines):
        lower = line.lower()
        for label in labels:
            ll = label.lower()
            if lower.startswith(ll + ":"):
                return line.split(":", 1)[1].strip()
            if lower == ll and idx + 1 < len(lines):
                return lines[idx + 1]
    return ""


def extract_committee(container: Tag) -> List[str]:
    committee_items: List[str] = []
    heading = None
    for h in container.find_all(["h2", "h3", "h4"]):
        if "committee" in h.get_text(" ", strip=True).lower():
            heading = h
            break
    if heading:
        for sib in heading.find_all_next(limit=30):
            if sib.name in {"h2", "h3", "h4"} and sib is not heading:
                break
            if sib.name in {"li", "p"}:
                t = normalize_whitespace(sib.get_text(" ", strip=True))
                if t:
                    committee_items.append(t)
    if not committee_items:
        text = normalize_whitespace(container.get_text("\n", strip=True))
        for line in text.splitlines():
            if re.search(r"\b(president|chair|secretary|treasurer|committee)\b", line, re.I):
                committee_items.append(line.strip())
    # Stable dedupe while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for item in committee_items:
        if item.lower() in seen:
            continue
        seen.add(item.lower())
        out.append(item)
    return out[:50]


def extract_record(url: str, html: str, scraped_at: str) -> SocietyRecord:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.h1:
        title = normalize_whitespace(soup.h1.get_text(" ", strip=True))
    if not title and soup.title:
        title = normalize_whitespace(soup.title.get_text(" ", strip=True))

    container = best_content_container(soup)
    if not container:
        container = soup.body if soup.body else soup
    remove_noise(container)

    chunks = extract_heading_and_bullets(container)
    raw_text = dedupe_lines("\n".join(chunks))
    fallback_text = dedupe_lines(normalize_whitespace(container.get_text("\n", strip=True)))
    if len(raw_text) < 80:
        raw_text = fallback_text

    email, phone = extract_emails_phones(raw_text)
    categories = extract_categories(soup, container)
    socials = extract_social_links(container, url)
    committee = extract_committee(container)
    membership_info = extract_field_by_label(raw_text, ("membership", "how to join", "join"))
    meeting_times = extract_field_by_label(raw_text, ("meeting times", "when we meet", "meetings"))
    last_updated = extract_field_by_label(raw_text, ("last updated", "updated", "last modified"))

    description = ""
    # Prefer first descriptive paragraph after title.
    paragraphs = [normalize_whitespace(p.get_text(" ", strip=True)) for p in container.find_all("p")]
    for p in paragraphs:
        if len(p) > 60:
            description = p
            break
    if not description:
        description = normalize_whitespace(raw_text.splitlines()[0] if raw_text else "")

    canonical = soup.find("link", attrs={"rel": lambda v: v and "canonical" in v})
    canonical_url = normalize_url(urljoin(url, canonical["href"])) if canonical and canonical.get("href") else normalize_url(url)

    record = SocietyRecord(
        name=title,
        url=canonical_url,
        description=description,
        categories=categories,
        contact={"email": email, "phone": phone},
        social_links=socials,
        committee=committee,
        membership_info=membership_info,
        meeting_times=meeting_times,
        last_updated=last_updated,
        raw_text=raw_text,
        scraped_at=scraped_at,
        sources={
            "name": canonical_url,
            "description": canonical_url,
            "categories": canonical_url,
            "contact": canonical_url,
            "social_links": canonical_url,
            "committee": canonical_url,
            "membership_info": canonical_url,
            "meeting_times": canonical_url,
            "last_updated": canonical_url if last_updated else "",
        },
    )
    return record


def crawl_society_urls(
    session: requests.Session,
    start_url: str,
    allowed_host: str,
    timeout: float,
    delay: float,
    max_pages: int,
    robots: Optional[robotparser.RobotFileParser],
    user_agent: str,
) -> Tuple[List[str], List[str]]:
    queue: List[str] = [normalize_url(start_url)]
    visited: Set[str] = set()
    society_urls: Set[str] = set()
    errors: List[str] = []
    state = {"last_request_time": 0.0}

    while queue and len(visited) < max_pages:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if robots and not (robots.can_fetch(user_agent, current) and robots.can_fetch("*", current)):
            logging.info("robots disallow: %s", current)
            continue

        html = fetch_page(session, current, timeout=timeout, delay=delay, state=state)
        if html is None:
            errors.append(current)
            continue
        if is_probably_js_rendered(html):
            logging.info("Possible JS-rendered page detected; trying Playwright: %s", current)
            rendered = fetch_with_playwright(current)
            if rendered:
                html = rendered

        found_societies, found_pages, snippets = extract_listing_candidates(html, current, allowed_host)
        if snippets:
            logging.info("Selector derivation snippets from %s: %s", current, " | ".join(snippets[:5]))
        for s in found_societies:
            if s not in visited:
                society_urls.add(s)
        for nxt in found_pages:
            if nxt not in visited and nxt not in queue and in_scope(nxt, allowed_host):
                queue.append(nxt)
        # Also discover more candidate listing pages in-scope with societies path.
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(BAD_SCHEMES):
                continue
            absolute = normalize_url(urljoin(current, href))
            if not in_scope(absolute, allowed_host):
                continue
            if "/get-involved/societies" in absolute.lower() and absolute not in visited and absolute not in queue:
                queue.append(absolute)

        logging.info(
            "Crawl progress: visited=%d queue=%d discovered_societies=%d",
            len(visited),
            len(queue),
            len(society_urls),
        )
    return sorted(society_urls), errors


def write_outputs(records: List[SocietyRecord]) -> None:
    with open(JSONL_FILE, "w", encoding="utf-8") as jf:
        for rec in records:
            jf.write(json.dumps(rec.as_json(), ensure_ascii=False) + "\n")
    with open(TEXT_FILE, "w", encoding="utf-8") as tf:
        for idx, rec in enumerate(records):
            entry = f"Name: {rec.name}\nURL: {rec.url}\n\n{rec.raw_text}".strip()
            tf.write(entry)
            if idx != len(records) - 1:
                tf.write("\n\n---\n\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape HW Union societies pages for RAG ingestion.")
    parser.add_argument("--start-url", default=BASE_LISTING_URL)
    parser.add_argument("--user-agent", default=DEFAULT_UA)
    parser.add_argument("--delay", type=float, default=1.0, help="Rate limit between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--max-pages", type=int, default=500, help="Safety cap on crawled pages.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    for path in (LOG_FILE, JSONL_FILE, TEXT_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging()
    scraped_at = datetime.now(timezone.utc).isoformat()
    start_url = normalize_url(args.start_url)
    host = urlparse(start_url).netloc.lower()

    logging.info("Starting scrape from %s", start_url)
    allowed, robots = parse_robots(start_url, args.user_agent, args.timeout)
    if not allowed:
        logging.error("robots.txt disallows scraping %s. Stopping.", start_url)
        print(
            "Stopped: robots.txt disallows scraping the start URL. "
            "Check scraper.log for details.",
            file=sys.stderr,
        )
        return 2

    session = make_session(args.user_agent)
    society_urls, crawl_errors = crawl_society_urls(
        session=session,
        start_url=start_url,
        allowed_host=host,
        timeout=args.timeout,
        delay=args.delay,
        max_pages=args.max_pages,
        robots=robots,
        user_agent=args.user_agent,
    )
    if not society_urls:
        logging.warning("No society pages discovered from %s", start_url)

    records: List[SocietyRecord] = []
    state = {"last_request_time": 0.0}
    visited: Set[str] = set()
    for url in society_urls:
        if url in visited:
            continue
        visited.add(url)
        if robots and not (robots.can_fetch(args.user_agent, url) and robots.can_fetch("*", url)):
            logging.info("Skipping disallowed society URL: %s", url)
            continue
        html = fetch_page(session, url, timeout=args.timeout, delay=args.delay, state=state)
        if html is None:
            logging.error("Skipping %s due to fetch failure", url)
            continue
        if is_probably_js_rendered(html):
            logging.info("JS-rendered detail page suspected; trying Playwright: %s", url)
            rendered = fetch_with_playwright(url)
            if rendered:
                html = rendered
        record = extract_record(url, html, scraped_at=scraped_at)
        if not record.name:
            logging.info("Skipping likely non-detail URL (no title): %s", url)
            continue
        records.append(record)
        logging.info("Extracted %s", record.name)

    write_outputs(records)
    logging.info(
        "Finished scrape. societies=%d crawl_errors=%d outputs=[%s,%s]",
        len(records),
        len(crawl_errors),
        JSONL_FILE,
        TEXT_FILE,
    )
    print(f"Wrote {len(records)} records to {JSONL_FILE} and {TEXT_FILE}.")
    if crawl_errors:
        print(f"Encountered {len(crawl_errors)} crawl fetch errors; see {LOG_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
