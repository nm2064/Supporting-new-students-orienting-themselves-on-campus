"""Collect the student-facing knowledge corpus from the university's public websites.

``KnowledgeCollector.collect()`` is the one entry point. It discovers pages from
sitemaps and seed URLs, keeps only pages inside the configured scope, fetches
them politely (robots.txt, a delay between requests), extracts the main content,
and returns a corpus in the ``{"pages": [...]}`` shape the RAG ingester reads.

Nothing is dropped for scoring low on keywords: an in-scope page with readable
content is kept. That is what lost pages such as the library opening hours from
the earlier pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup, Tag


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    content_type: str
    text: str


Fetch = Callable[[str], FetchResult]

DEFAULT_USER_AGENT = "UniBotPageExtractor/2.0 (+https://uni-bot.duckdns.org; campus-information-refresh)"


@dataclass(frozen=True)
class Scope:
    """Which pages belong in the corpus: path prefixes per host, minus exclusions."""

    include: dict[str, list[str]]
    exclude: list[str] = field(default_factory=list)

    def contains(self, url: str) -> bool:
        parts = urlparse(url)
        if parts.scheme not in {"http", "https"}:
            return False
        prefixes = self.include.get(parts.netloc)
        if prefixes is None:
            return False
        path = parts.path or "/"
        if any(_under(path, prefix) for prefix in self.exclude):
            return False
        return any(_under(path, prefix) for prefix in prefixes)


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return prefix == "" or path == prefix or path.startswith(prefix + "/")


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication: https, no fragment, query or trailing slash."""
    parts = urlparse(url.strip())
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/") or "/"
    scheme = "https" if parts.scheme.lower() in {"http", "https"} else parts.scheme.lower()
    return urlunparse((scheme, parts.netloc.lower(), path, "", "", ""))


# ---- content extraction --------------------------------------------------------

_NOISE_SELECTORS = [
    # Not "form": some sites (the Students' Union's ASP.NET pages) wrap the whole
    # page in one <form>. Remove the controls and search forms instead.
    "script", "style", "noscript", "svg", "iframe", "nav", "header", "footer",
    "input", "select", "textarea", "button", "form[role='search']", "form.search",
    "[role='navigation']", "[aria-label='breadcrumb']", ".breadcrumb", ".breadcrumbs",
    ".cookie", "#cookie", ".skip-link", "[data-related-content]",
]
_CONTENT_SELECTORS = ["main", "article", "[role='main']", ".entry-content", ".post-content", ".content"]
_SITE_SUFFIX = re.compile(r"\s*[|–—-]\s*(Heriot-Watt University.*|HWU.*)$", re.IGNORECASE)
_SKIP_HEADINGS = {"related content", "share this page", "explore more"}


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    # get_text(" ") puts a space before punctuation that follows inline tags.
    return re.sub(r" ([.,;:!?)])", r"\1", text)


def _content_root(soup: BeautifulSoup) -> Tag | None:
    for selector in _CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node and len(node.get_text(" ", strip=True)) > 80:
            return node
    return soup.body


def extract_page(html: str) -> tuple[str, list[str], str]:
    """Return (title, headings, text) for the main content of an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    root = _content_root(soup)
    if root is None:
        return "", [], ""
    for selector in _NOISE_SELECTORS:
        for node in root.select(selector):
            node.decompose()

    h1 = root.find("h1")
    title = _clean(h1.get_text(" ")) if h1 else ""
    if not title and soup.title:
        title = _SITE_SUFFIX.sub("", _clean(soup.title.get_text(" ")))

    lines: list[str] = []
    headings: list[str] = []
    seen: set[str] = set()
    skipping = False
    for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if node.name != "table" and node.find_parent("table") is not None:
            continue
        if node.name != "li" and node.find_parent("li") is not None:
            continue
        text = _table_text(node) if node.name == "table" else _clean(node.get_text(" "))
        if not text:
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            skipping = text.lower() in _SKIP_HEADINGS
            if skipping:
                continue
            if text not in headings:
                headings.append(text)
            line = f"\n{text}"
        elif skipping:
            continue
        elif node.name == "li":
            line = f"- {text}"
        elif node.name == "table":
            line = text
        else:
            line = text
        if line.strip().lower() in seen:
            continue
        seen.add(line.strip().lower())
        lines.append(line)
    return title, headings, "\n".join(lines).strip()


def _table_text(table: Tag) -> str:
    """One line per row, each cell labelled with its column header when known."""
    rows = [
        [_clean(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
        for row in table.find_all("tr")
    ]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    header = rows[0] if table.find("tr").find("th") is not None else None
    body = rows[1:] if header else rows
    lines = []
    for row in body:
        if header and len(header) == len(row):
            lines.append(" | ".join(f"{h}: {c}" if h else c for h, c in zip(header, row)))
        else:
            lines.append(" | ".join(row))
    return "\n".join(lines)


def _links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]


def _sitemap_locations(xml: str) -> list[str]:
    return [_clean(loc) for loc in re.findall(r"<loc>\s*([^<]+?)\s*</loc>", xml)]


# ---- collection ----------------------------------------------------------------


class KnowledgeCollector:
    MIN_TEXT_CHARS = 80

    def __init__(
        self,
        fetch: Fetch,
        scope: Scope,
        delay_s: float = 1.0,
        max_pages: int = 3000,
        site_labels: dict[str, str] | None = None,
        max_link_depth: int = 2,
        user_agent: str = DEFAULT_USER_AGENT,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.fetch = fetch
        self._clock = clock
        self._sleep = sleep
        self._last_request: dict[str, float] = {}
        self.scope = scope
        self.delay_s = delay_s
        self.max_pages = max_pages
        self.site_labels = site_labels or {}
        # How many link hops from a sitemap/seed page to follow (in scope only).
        # Sitemaps miss pages, e.g. the Edinburgh library opening hours.
        self.max_link_depth = max_link_depth
        self.user_agent = user_agent
        self._robots: dict[str, RobotFileParser] = {}

    def _allowed(self, url: str) -> bool:
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser = RobotFileParser()
            result = self._get(f"{origin}/robots.txt")
            # A missing robots.txt allows everything; an unreachable one allows nothing.
            if result is None or result.status >= 500:
                parser.parse(["User-agent: *", "Disallow: /"])
            else:
                parser.parse(result.text.splitlines() if result.status == 200 else [])
            self._robots[origin] = parser
        return self._robots[origin].can_fetch(self.user_agent, url)

    def _get(self, url: str) -> FetchResult | None:
        """Fetch politely: each host waits max(delay_s, its robots Crawl-delay)."""
        parts = urlparse(url)
        host = parts.netloc
        robots = self._robots.get(f"{parts.scheme}://{host}")
        gap = max(self.delay_s, float((robots.crawl_delay(self.user_agent) if robots else None) or 0))
        last = self._last_request.get(host)
        if last is not None:
            wait = gap - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()
        try:
            return self.fetch(url)
        except Exception:
            return None

    def collect(
        self,
        sitemaps: Iterable[str] = (),
        seed_urls: Iterable[str] = (),
        library_hours: LibraryHoursSource | None = None,
        student_events: StudentEventsSource | None = None,
        societies: SocietiesSource | None = None,
    ) -> dict:
        queue: list[tuple[str, int]] = []
        queued: set[str] = set()

        def enqueue(url: str, depth: int = 0) -> None:
            url = normalize_url(url)
            if url not in queued and self.scope.contains(url):
                queued.add(url)
                if self._allowed(url):
                    queue.append((url, depth))

        for sitemap_url in sitemaps:
            result = self._get(sitemap_url)
            if result and result.status == 200:
                for location in _sitemap_locations(result.text):
                    enqueue(location)
        for url in seed_urls:
            enqueue(url)

        pages: list[dict] = []
        unavailable: list[dict] = []
        seen_content: set[str] = set()
        while queue and len(pages) < self.max_pages:
            url, depth = queue.pop(0)
            result = self._get(url)
            if result is None or result.status != 200 or "html" not in result.content_type:
                unavailable.append({"url": url, "status": result.status if result else None})
                continue
            if depth < self.max_link_depth:
                for link in _links(result.text, url):
                    enqueue(link, depth + 1)
            title, headings, text = extract_page(result.text)
            if len(text) < self.MIN_TEXT_CHARS:
                unavailable.append({"url": url, "status": "no readable content"})
                continue
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in seen_content:
                continue  # the same page published at another address
            seen_content.add(fingerprint)
            host = urlparse(url).netloc
            pages.append(
                {
                    "title": title,
                    "url": url,
                    "category": _category(url),
                    "site_label": self.site_labels.get(host, host),
                    "headings": headings,
                    "extracted_text": text,
                }
            )

        if library_hours is not None:
            result = self._get(library_hours.api_url) if self._allowed(library_hours.api_url) else None
            try:
                payload = json.loads(result.text) if result and result.status == 200 else None
            except json.JSONDecodeError:
                payload = None
            if payload:
                pages.extend(_library_hours_pages(payload, library_hours))
            else:
                unavailable.append({"url": library_hours.api_url, "status": result.status if result else None})

        if student_events is not None:
            result = self._get(student_events.api_url) if self._allowed(student_events.api_url) else None
            if result and result.status == 200:
                for page in _student_event_pages(result.text):
                    event = page["event"]
                    if not event["start"] or not event["end"] or event["end"] < event["start"]:
                        unavailable.append({"url": page["url"], "reason": "Invalid event start or end time"})
                    else:
                        pages.append(page)
            else:
                unavailable.append({"url": student_events.api_url, "status": result.status if result else None})

        if societies is not None:
            result = self._get(societies.api_url) if self._allowed(societies.api_url) else None
            try:
                groups = json.loads(result.text) if result and result.status == 200 else None
            except json.JSONDecodeError:
                groups = None
            if isinstance(groups, list):
                society_pages = _society_pages(groups, societies)
                # These replace any HTML copies of the same pages.
                replaced = {page["url"] for page in society_pages}
                pages[:] = [page for page in pages if page["url"] not in replaced] + society_pages
            else:
                unavailable.append({"url": societies.api_url, "status": result.status if result else None})
        return {"pages": pages, "unavailable": unavailable}


@dataclass(frozen=True)
class LibraryHoursSource:
    """Opening hours from the library's LibCal hours API.

    The university pages load their hours tables with JavaScript, so the HTML
    alone never contains the times. ``pages`` maps a top-level LibCal location id
    to the public page to cite for it; others cite ``default_page``.
    """

    api_url: str
    pages: dict[int, str]
    default_page: str


def _library_hours_pages(payload: dict, source: LibraryHoursSource) -> list[dict]:
    from datetime import date

    locations = payload.get("locations") or list(payload.values())
    groups: dict[int, list[dict]] = {}
    names: dict[int, str] = {}
    for location in locations:
        if not isinstance(location, dict) or "lid" not in location:
            continue
        root = location.get("parent_lid") or location["lid"]
        groups.setdefault(root, []).append(location)
        if not location.get("parent_lid"):
            names[root] = location.get("name", "Library")

    pages = []
    for root, members in groups.items():
        lines = [
            "Opening hours from the library's live timetable. Hours change during vacations "
            "and holidays; check the linked page for the latest times."
        ]
        for location in members:
            for week in location.get("weeks", []):
                for day_name, day in week.items():
                    if not isinstance(day, dict) or not day.get("date"):
                        continue
                    when = date.fromisoformat(day["date"])
                    rendered = day.get("rendered") or "No hours published"
                    lines.append(
                        f"{location.get('name', '')}: {day_name} {when.day} {when:%B %Y}: {rendered}"
                    )
        name = names.get(root, members[0].get("name", "Library"))
        pages.append(
            {
                "title": f"{name} opening hours",
                "url": source.pages.get(root, source.default_page),
                "category": "library/opening-hours",
                "site_label": "Heriot-Watt University Library",
                "headings": [f"{name} opening hours"],
                "extracted_text": "\n".join(lines),
            }
        )
    return pages


@dataclass(frozen=True)
class StudentEventsSource:
    """Upcoming student events from the LibCal events feed (one HTML table per event)."""

    api_url: str


def _student_event_pages(feed_html: str) -> list[dict]:
    soup = BeautifulSoup(feed_html, "html.parser")
    pages = []
    for table in soup.find_all("table"):
        fields: dict[str, str] = {}
        link = ""
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            label = _clean(cells[0].get_text(" ")).rstrip(":")
            fields[label] = _clean(cells[1].get_text(" "))
            anchor = cells[1].find("a", href=True)
            if label == "Title" and anchor:
                link = normalize_url(anchor["href"])
        title = fields.pop("Title", "")
        if not title or not link:
            continue
        lines = [f"{title} is a Heriot-Watt student event."]
        start, end = fields.pop("From", ""), fields.pop("To", "")
        if start:
            lines.append(f"When: {start}" + (f" to {end}" if end else ""))
        location = fields.pop("Location", "")
        if location:
            lines.append(f"Where: {location}")
        lines.extend(f"{label}: {value}" for label, value in fields.items() if value)
        pages.append(
            {
                "title": f"{title} (student event)",
                "url": link,
                "category": "student-events",
                "site_label": "Heriot-Watt Student Events",
                "headings": [title],
                "extracted_text": "\n".join(lines),
                # Structured copy for the Events page.
                "event": {
                    "title": title,
                    "start": _libcal_time(start),
                    "end": _libcal_time(end),
                    "location": location,
                    "campus": _campus_named_in(" ".join([location, *fields.values()])),
                    "url": link,
                },
            }
        )
    return pages


_CAMPUSES = {
    "edinburgh": "edinburgh",
    "scottish borders": "scottish-borders",
    "borders": "scottish-borders",
    "orkney": "orkney",
    "dubai": "dubai",
    "malaysia": "malaysia",
}


def _campus_named_in(text: str) -> str | None:
    lowered = text.lower()
    for name, campus in _CAMPUSES.items():
        if name in lowered:
            return campus
    return None


def _libcal_time(value: str) -> str | None:
    """'11:00 Wednesday, November 11, 2026' -> '2026-11-11T11:00:00' (campus local time)."""
    from datetime import datetime

    for pattern in ("%H:%M %A, %B %d, %Y", "%A, %B %d, %Y"):
        try:
            return datetime.strptime(value.strip(), pattern).isoformat()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class SocietiesSource:
    """Every Students' Union society, from the JSON feed behind the societies directory.

    The directory page builds its list with JavaScript, so its HTML has no societies.
    """

    api_url: str
    directory_page: str


def _html_to_text(fragment: str) -> str:
    soup = BeautifulSoup(fragment or "", "html.parser")
    blocks = [_clean(node.get_text(" ")) for node in soup.find_all(["p", "li", "h1", "h2", "h3", "h4"])]
    text = "\n".join(block for block in blocks if block)
    return text or _clean(soup.get_text(" "))


def _society_pages(groups: list, source: SocietiesSource) -> list[dict]:
    pages = []
    directory: dict[str, list[str]] = {}
    for group in groups:
        name = _clean(str(group.get("Name") or ""))
        url = group.get("Url") or ""
        if not name or not url:
            continue
        category = _clean(str(group.get("Category") or "Other"))
        directory.setdefault(category, []).append(name)
        article = "an" if category[:1].lower() in "aeiou" else "a"
        lines = [f"{name} is {article} {category} society at the Heriot-Watt Students' Union."]
        statuses = [v for a in group.get("Attributes") or [] if a.get("Set") == "Status" for v in a.get("Values", [])]
        if statuses:
            lines.append(f"Status: {', '.join(statuses)}")
        if group.get("EmailAddress"):
            lines.append(f"Contact: {group['EmailAddress']}")
        if group.get("InstagramUsername"):
            lines.append(f"Instagram: @{group['InstagramUsername'].lstrip('@')}")
        for label, key in (("Facebook", "FacebookUrl"), ("Website", "WebsiteUrl")):
            if group.get(key):
                lines.append(f"{label}: {group[key]}")
        description = _html_to_text(group.get("LongDescription") or group.get("Description") or "")
        if description:
            lines.append(description)
        pages.append(
            {
                "title": f"{name} (society)",
                "url": normalize_url(url),
                "category": f"societies/{category.lower()}",
                "site_label": "Heriot-Watt Students' Union",
                "headings": [name],
                "extracted_text": "\n".join(lines),
            }
        )
    if directory:
        lines = [
            "All societies at the Heriot-Watt Students' Union, by category. Each society has its own "
            "page with contact details and how to join."
        ]
        for category in sorted(directory):
            lines.append(f"{category}: {', '.join(sorted(directory[category]))}")
        pages.append(
            {
                "title": "Students' Union societies directory",
                "url": normalize_url(source.directory_page),
                "category": "societies",
                "site_label": "Heriot-Watt Students' Union",
                "headings": ["Societies directory"],
                "extracted_text": "\n".join(lines),
            }
        )
    return pages


def _category(url: str) -> str:
    segments = [s for s in urlparse(url).path.split("/") if s]
    return "/".join(segments[:2]) if segments else "home"


# ---- production configuration ----------------------------------------------------

HWU = "https://www.hw.ac.uk"
LIBRARY_HOURS_PAGE = f"{HWU}/about/professional-services/information-services/library-essentials/library-opening-hours"

PRODUCTION_SCOPE = Scope(
    include={
        "www.hw.ac.uk": [
            "/students",
            "/campuses",
            "/about/professional-services/information-services",
            "/about/professional-services/registry-services",
            "/about/professional-services/campus-services",
            "/about/professional-services/finance",
            "/study/fees-and-funding",
            "/study/how-to-apply",
        ],
        "www.hwunion.com": [
            "/advice-hub", "/societies", "/groups", "/get-involved", "/about-us", "/your-union",
            "/volunteer", "/be-represented", "/book-a-room", "/sustainability",
        ],
        "sportsunion.site.hw.ac.uk": [""],
        "halls.wattliving.co.uk": [""],
    },
    exclude=[
        "/profiles", "/news", "/search", "/global-search", "/document-library", "/testing",
        # blog/archive listings on the partner sites
        "/story", "/tag", "/category", "/author", "/feed", "/wp-content", "/wp-json", "/event",
        "/2019", "/2020", "/2021", "/2022", "/2023", "/2024", "/2025", "/2026", "/2027",
    ],
)

PRODUCTION_SITEMAPS = [
    f"{HWU}/sitemap.xml",
    "https://sportsunion.site.hw.ac.uk/wp-sitemap-posts-page-1.xml",
]

PRODUCTION_SEEDS = [
    LIBRARY_HOURS_PAGE,
    "https://www.hwunion.com/advice-hub",
    "https://www.hwunion.com/societies",
    "https://www.hwunion.com/get-involved",
    "https://www.hwunion.com/about-us",
    "https://halls.wattliving.co.uk/",
]

PRODUCTION_SITE_LABELS = {
    "www.hw.ac.uk": "Heriot-Watt University",
    "www.hwunion.com": "Heriot-Watt Students' Union",
    "sportsunion.site.hw.ac.uk": "Heriot-Watt Sports Union",
    "halls.wattliving.co.uk": "WattLiving (Heriot-Watt halls)",
}

PRODUCTION_LIBRARY_HOURS = LibraryHoursSource(
    api_url="https://hw.ac.libcal.com/api_hours_grid.php?iid=854&format=json&weeks=2&systemTime=0",
    pages={
        635: f"{LIBRARY_HOURS_PAGE}/edinburgh",
        2569: f"{LIBRARY_HOURS_PAGE}/scottish-borders",
        2585: f"{LIBRARY_HOURS_PAGE}/malaysia",
    },
    default_page=LIBRARY_HOURS_PAGE,
)

PRODUCTION_STUDENT_EVENTS = StudentEventsSource(
    api_url="https://hw.ac.libcal.com/api_events.php?m=month&cid=9127&audience=&c=&d=&tar=0&context=object&format=json",
)


PRODUCTION_SOCIETIES = SocietiesSource(
    api_url="https://www.hwunion.com/svc/feeds/groups/6033?subtree=true&attributesets=Status",
    directory_page="https://www.hwunion.com/get-involved/societies",
)


def http_fetch(user_agent: str, timeout_s: float = 20.0) -> Fetch:
    """A real fetcher: retries transient failures, identifies itself, follows redirects."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "en-GB,en;q=0.9"})

    def fetch(url: str) -> FetchResult:
        response = session.get(url, timeout=timeout_s)
        return FetchResult(
            url=response.url,
            status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            text=response.text,
        )

    return fetch


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Collect the UniBot knowledge corpus.")
    parser.add_argument("--out", required=True, help="Where to write the corpus JSON")
    parser.add_argument("--delay", type=float, default=1.0, help="Minimum seconds between requests to a host")
    parser.add_argument("--max-pages", type=int, default=3000)
    args = parser.parse_args(argv)

    collector = KnowledgeCollector(
        fetch=http_fetch(DEFAULT_USER_AGENT),
        scope=PRODUCTION_SCOPE,
        delay_s=args.delay,
        max_pages=args.max_pages,
        site_labels=PRODUCTION_SITE_LABELS,
    )
    started = datetime.now(timezone.utc)
    corpus = collector.collect(
        sitemaps=PRODUCTION_SITEMAPS,
        seed_urls=PRODUCTION_SEEDS,
        library_hours=PRODUCTION_LIBRARY_HOURS,
        student_events=PRODUCTION_STUDENT_EVENTS,
        societies=PRODUCTION_SOCIETIES,
    )
    corpus["collected_at"] = started.isoformat()
    corpus["total_pages"] = len(corpus["pages"])
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(corpus, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"pages": len(corpus["pages"]), "unavailable": len(corpus["unavailable"]), "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
