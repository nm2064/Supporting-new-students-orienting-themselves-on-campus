#!/usr/bin/env python3
"""Crawl Heriot-Watt related sites and extract useful internal links.

Default presets cover:
- Heriot-Watt University: hw.ac.uk
- Watt Living: reslife.wattliving.co.uk
- Heriot-Watt Student Union: hwunion.com
- Heriot-Watt Sports Union: sportsunion.site.hw.ac.uk

Usage:
  python crawl_hw_website.py
  python crawl_hw_website.py --sites all --max-pages-per-site 150
  python crawl_hw_website.py --sites hwu watt-living union sports-union
  python crawl_hw_website.py --sites hwu --output-prefix hwu_links
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_JSONL = Path("hw_useful_links.jsonl")
DEFAULT_TEXT = Path("hw_useful_links.txt")
DEFAULT_LOG = Path("hw_website_crawler.log")
DEFAULT_USER_AGENT = (
    "UniBotWebsiteCrawler/1.0 (+https://example.invalid/contact; purpose=academic-research)"
)
BAD_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "#")
TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}
SKIP_PATH_HINTS = (
    "/search",
    "/media/",
    "/assets/",
    "/profiles/",
    "/fonts/",
    "/scripts/",
    "/css/",
    "/js/",
    "/wp-content/",
    "/wp-admin/",
    "/api/",
    "/feed",
    "/cdn-cgi/",
)
SKIP_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".xml",
    ".zip",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".wmv",
}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
USEFUL_KEYWORDS = {
    "student",
    "students",
    "campus",
    "library",
    "course",
    "programme",
    "program",
    "module",
    "fees",
    "funding",
    "scholarship",
    "accommodation",
    "housing",
    "watt living",
    "visa",
    "international",
    "support",
    "service",
    "wellbeing",
    "health",
    "counselling",
    "careers",
    "career",
    "registry",
    "enrolment",
    "enrollment",
    "timetable",
    "exam",
    "assessment",
    "graduation",
    "policies",
    "policy",
    "union",
    "student union",
    "sports union",
    "society",
    "societies",
    "sport",
    "events",
    "travel",
    "transport",
    "parking",
    "it",
    "digital",
    "help",
    "contact",
    "about",
    "study",
    "reslife",
    "residential",
    "residence",
}
LOW_VALUE_KEYWORDS = {
    "privacy",
    "cookie",
    "cookies",
    "legal",
    "accessibility",
    "terms",
    "press",
    "news",
    "vacancy",
    "vacancies",
    "job",
    "jobs",
}
SITE_PRESETS = {
    "hwu": {
        "label": "Heriot-Watt University",
        "start_url": "https://www.hw.ac.uk/",
        "allowed_hosts": ["hw.ac.uk"],
        "include_subdomains": True,
    },
    "watt-living": {
        "label": "Watt Living",
        "start_url": "https://reslife.wattliving.co.uk/",
        "allowed_hosts": ["reslife.wattliving.co.uk", "wattliving.co.uk"],
        "include_subdomains": True,
    },
    "union": {
        "label": "Heriot-Watt Student Union",
        "start_url": "https://www.hwunion.com/",
        "allowed_hosts": ["hwunion.com"],
        "include_subdomains": True,
    },
    "sports-union": {
        "label": "Heriot-Watt Sports Union",
        "start_url": "https://sportsunion.site.hw.ac.uk/",
        "allowed_hosts": ["sportsunion.site.hw.ac.uk", "site.hw.ac.uk"],
        "include_subdomains": True,
    },
}


@dataclass(slots=True)
class SiteConfig:
    key: str
    label: str
    start_url: str
    allowed_hosts: tuple[str, ...]
    include_subdomains: bool


@dataclass(slots=True)
class CrawlConfig:
    sites: list[SiteConfig]
    max_pages_per_site: int
    delay: float
    timeout: float
    user_agent: str
    output_jsonl: Path
    output_text: Path
    log_file: Path
    min_usefulness_score: int


@dataclass(slots=True)
class LinkRecord:
    site_key: str
    site_label: str
    url: str
    source_url: str
    anchor_text: str
    title: str
    description: str
    headings: list[str] = field(default_factory=list)
    content_type: str = ""
    usefulness_score: int = 0
    usefulness_reasons: list[str] = field(default_factory=list)
    is_document: bool = False
    scraped_at: str = ""


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


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    filtered_query = sorted(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in TRACKING_QUERY_KEYS
    )
    query = urlencode(filtered_query, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def host_matches(host: str, allowed_host: str, include_subdomains: bool) -> bool:
    if include_subdomains:
        return host == allowed_host or host.endswith("." + allowed_host)
    return host == allowed_host


def in_scope(url: str, allowed_hosts: Sequence[str], include_subdomains: bool) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    return any(host_matches(host, allowed_host, include_subdomains) for allowed_host in allowed_hosts)


def should_skip_url(url: str) -> bool:
    lower = url.lower()
    if any(lower.startswith(prefix) for prefix in BAD_SCHEMES):
        return True
    if any(hint in lower for hint in SKIP_PATH_HINTS):
        return True
    path = urlparse(lower).path
    suffix = Path(path).suffix
    return suffix in SKIP_EXTENSIONS


def is_document_url(url: str) -> bool:
    return Path(urlparse(url).path.lower()).suffix in DOCUMENT_EXTENSIONS


def parse_robots(site: SiteConfig, user_agent: str, timeout: float) -> robotparser.RobotFileParser | None:
    parsed = urlparse(site.start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        response = requests.get(
            robots_url,
            headers={"User-Agent": user_agent},
            timeout=timeout,
        )
        if response.status_code >= 400:
            logging.warning("robots.txt returned %s at %s", response.status_code, robots_url)
            return None
        rp.parse(response.text.splitlines())
        return rp
    except requests.RequestException as exc:
        logging.warning("Failed to load robots.txt from %s: %s", robots_url, exc)
        return None


def robots_allow(rp: robotparser.RobotFileParser | None, user_agent: str, url: str) -> bool:
    if rp is None:
        return True
    return rp.can_fetch(user_agent, url) and rp.can_fetch("*", url)


def fetch_url(
    session: requests.Session,
    url: str,
    timeout: float,
    delay: float,
    state: dict[str, float],
) -> requests.Response | None:
    elapsed = time.monotonic() - state.get("last_request_time", 0.0)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    try:
        response = session.get(url, timeout=timeout)
        state["last_request_time"] = time.monotonic()
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        logging.warning("Request failed for %s: %s", url, exc)
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
        ".navigation",
        ".breadcrumb",
        ".breadcrumbs",
        ".share",
        ".social-share",
        ".cookie",
        ".consent",
        ".sidebar",
        ".hero-video",
    ):
        for node in container.select(selector):
            node.decompose()


def collapse_text(parts: Iterable[str]) -> str:
    lines: list[str] = []
    for item in parts:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned:
            lines.append(cleaned)
    return " ".join(lines)


def page_description(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        return collapse_text([str(meta["content"])])
    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("main p, article p, p")[:3]]
    return collapse_text(paragraphs)


def page_headings(soup: BeautifulSoup) -> list[str]:
    headings: list[str] = []
    for heading in soup.select("main h1, main h2, article h1, article h2, h1, h2")[:6]:
        text = collapse_text([heading.get_text(" ", strip=True)])
        if text:
            headings.append(text)
    return headings


def useful_score(url: str, title: str, description: str, headings: Sequence[str]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    haystack = " ".join([url, title, description, *headings]).lower()

    if is_document_url(url):
        score += 2
        reasons.append("downloadable document")

    keyword_hits = sorted(keyword for keyword in USEFUL_KEYWORDS if keyword in haystack)
    if keyword_hits:
        score += min(6, len(keyword_hits))
        reasons.append(f"useful keywords: {', '.join(keyword_hits[:6])}")

    if "/students" in haystack or "/student" in haystack:
        score += 3
        reasons.append("student-focused page")

    if title:
        score += 1
        reasons.append("has page title")

    if description:
        score += 1
        reasons.append("has page summary")

    low_value_hits = sorted(keyword for keyword in LOW_VALUE_KEYWORDS if keyword in haystack)
    if low_value_hits:
        score -= min(4, len(low_value_hits))
        reasons.append(f"low-value keywords: {', '.join(low_value_hits[:4])}")

    if len(description) < 40 and not is_document_url(url):
        score -= 1
        reasons.append("thin page summary")

    if "/news/" in haystack or "/press/" in haystack:
        score -= 2
        reasons.append("likely news or press content")

    return score, reasons


def extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    anchors: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", "")).strip()
        if not href or any(href.startswith(prefix) for prefix in BAD_SCHEMES):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        text = collapse_text([anchor.get_text(" ", strip=True)])
        anchors.append((absolute, text))
    return anchors


def build_record(
    site: SiteConfig,
    url: str,
    source_url: str,
    anchor_text: str,
    html: str | None,
    content_type: str,
) -> LinkRecord:
    title = ""
    description = ""
    headings: list[str] = []

    if html and "html" in content_type.lower():
        soup = BeautifulSoup(html, "html.parser")
        remove_noise(soup)
        title = collapse_text([soup.title.get_text(" ", strip=True)]) if soup.title else ""
        description = page_description(soup)
        headings = page_headings(soup)

    score, reasons = useful_score(url, title, description, headings)
    return LinkRecord(
        site_key=site.key,
        site_label=site.label,
        url=url,
        source_url=source_url,
        anchor_text=anchor_text,
        title=title,
        description=description,
        headings=headings,
        content_type=content_type,
        usefulness_score=score,
        usefulness_reasons=reasons,
        is_document=is_document_url(url),
        scraped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def crawl_site(
    site: SiteConfig,
    session: requests.Session,
    max_pages: int,
    timeout: float,
    delay: float,
    user_agent: str,
    min_usefulness_score: int,
) -> list[LinkRecord]:
    rp = parse_robots(site, user_agent, timeout)
    request_state: dict[str, float] = {"last_request_time": 0.0}
    queue = deque([normalize_url(site.start_url)])
    seen_pages: set[str] = set()
    discovered_records: dict[str, LinkRecord] = {}

    while queue and len(seen_pages) < max_pages:
        current_url = queue.popleft()
        if current_url in seen_pages:
            continue
        if should_skip_url(current_url):
            continue
        if not in_scope(current_url, site.allowed_hosts, site.include_subdomains):
            continue
        if not robots_allow(rp, user_agent, current_url):
            logging.info("Blocked by robots.txt: %s", current_url)
            continue

        response = fetch_url(session, current_url, timeout, delay, request_state)
        if response is None:
            continue

        content_type = response.headers.get("content-type", "")
        seen_pages.add(current_url)
        logging.info("[%s] Crawled %s (%s/%s)", site.key, current_url, len(seen_pages), max_pages)

        if "html" not in content_type.lower():
            record = build_record(
                site=site,
                url=current_url,
                source_url=current_url,
                anchor_text="",
                html=None,
                content_type=content_type,
            )
            discovered_records.setdefault(current_url, record)
            continue

        html = response.text
        page_record = build_record(
            site=site,
            url=current_url,
            source_url=current_url,
            anchor_text="",
            html=html,
            content_type=content_type,
        )
        existing = discovered_records.get(current_url)
        if existing is None or page_record.usefulness_score > existing.usefulness_score:
            discovered_records[current_url] = page_record

        for next_url, anchor_text in extract_links(html, current_url):
            if should_skip_url(next_url):
                continue
            if not in_scope(next_url, site.allowed_hosts, site.include_subdomains):
                continue

            if next_url not in discovered_records:
                discovered_records[next_url] = build_record(
                    site=site,
                    url=next_url,
                    source_url=current_url,
                    anchor_text=anchor_text,
                    html=None,
                    content_type="link",
                )
            elif anchor_text and not discovered_records[next_url].anchor_text:
                discovered_records[next_url].anchor_text = anchor_text

            if next_url not in seen_pages and not is_document_url(next_url):
                queue.append(next_url)

    useful_records = [
        record
        for record in discovered_records.values()
        if record.usefulness_score >= min_usefulness_score
    ]
    useful_records.sort(key=lambda item: (-item.usefulness_score, item.title.lower(), item.url.lower()))
    return useful_records


def crawl(config: CrawlConfig) -> list[LinkRecord]:
    session = make_session(config.user_agent)
    all_records: list[LinkRecord] = []
    for site in config.sites:
        logging.info(
            "Starting site crawl: key=%s label=%s start_url=%s hosts=%s max_pages=%s",
            site.key,
            site.label,
            site.start_url,
            ",".join(site.allowed_hosts),
            config.max_pages_per_site,
        )
        records = crawl_site(
            site=site,
            session=session,
            max_pages=config.max_pages_per_site,
            timeout=config.timeout,
            delay=config.delay,
            user_agent=config.user_agent,
            min_usefulness_score=config.min_usefulness_score,
        )
        logging.info("Finished site crawl: key=%s useful_links=%s", site.key, len(records))
        all_records.extend(records)

    all_records.sort(
        key=lambda item: (item.site_key.lower(), -item.usefulness_score, item.title.lower(), item.url.lower())
    )
    return all_records


def write_outputs(records: Sequence[LinkRecord], jsonl_path: Path, text_path: Path) -> None:
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with text_path.open("w", encoding="utf-8") as handle:
        current_site = None
        for record in records:
            if record.site_label != current_site:
                current_site = record.site_label
                handle.write(f"=== {current_site} ({record.site_key}) ===\n\n")
            title = record.title or record.anchor_text or record.url
            handle.write(f"{title}\n")
            handle.write(f"URL: {record.url}\n")
            handle.write(f"Source: {record.source_url}\n")
            handle.write(f"Score: {record.usefulness_score}\n")
            if record.description:
                handle.write(f"Summary: {record.description}\n")
            if record.headings:
                handle.write(f"Headings: {' | '.join(record.headings)}\n")
            if record.usefulness_reasons:
                handle.write(f"Why useful: {'; '.join(record.usefulness_reasons)}\n")
            handle.write("\n")


def resolve_sites(site_names: Sequence[str]) -> list[SiteConfig]:
    requested = list(site_names)
    if not requested or "all" in requested:
        requested = list(SITE_PRESETS.keys())

    resolved: list[SiteConfig] = []
    seen: set[str] = set()
    for name in requested:
        key = name.lower()
        if key == "all":
            continue
        preset = SITE_PRESETS.get(key)
        if preset is None:
            valid = ", ".join(["all", *SITE_PRESETS.keys()])
            raise SystemExit(f"Unknown site preset '{name}'. Valid values: {valid}")
        if key in seen:
            continue
        seen.add(key)
        resolved.append(
            SiteConfig(
                key=key,
                label=str(preset["label"]),
                start_url=normalize_url(str(preset["start_url"])),
                allowed_hosts=tuple(str(host).lower() for host in preset["allowed_hosts"]),
                include_subdomains=bool(preset["include_subdomains"]),
            )
        )
    return resolved


def parse_args(argv: Sequence[str]) -> CrawlConfig:
    parser = argparse.ArgumentParser(
        description="Crawl Heriot-Watt related sites and extract useful internal links."
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["all"],
        help="Site presets to crawl: all, hwu, watt-living, union, sports-union",
    )
    parser.add_argument("--max-pages-per-site", type=int, default=150)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--output-jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--output-text", default=str(DEFAULT_TEXT))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--min-usefulness-score", type=int, default=2)
    args = parser.parse_args(argv)

    output_jsonl = Path(args.output_jsonl)
    output_text = Path(args.output_text)
    log_file = Path(args.log_file)
    if args.output_prefix:
        prefix = Path(args.output_prefix)
        output_jsonl = Path(f"{prefix}.jsonl")
        output_text = Path(f"{prefix}.txt")
        log_file = Path(f"{prefix}.log")

    return CrawlConfig(
        sites=resolve_sites(args.sites),
        max_pages_per_site=max(1, args.max_pages_per_site),
        delay=max(0.0, args.delay),
        timeout=max(1.0, args.timeout),
        user_agent=args.user_agent,
        output_jsonl=output_jsonl,
        output_text=output_text,
        log_file=log_file,
        min_usefulness_score=args.min_usefulness_score,
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    configure_logging(config.log_file)
    logging.info(
        "Starting crawl for sites=%s max_pages_per_site=%s",
        ",".join(site.key for site in config.sites),
        config.max_pages_per_site,
    )
    records = crawl(config)
    write_outputs(records, config.output_jsonl, config.output_text)
    logging.info("Finished crawl with %s useful links", len(records))
    print(f"Wrote {len(records)} useful links to {config.output_jsonl} and {config.output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
