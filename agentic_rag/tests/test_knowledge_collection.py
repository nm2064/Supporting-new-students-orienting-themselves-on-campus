"""Seam: KnowledgeCollector.collect() against a fake web (HTTP is the boundary)."""

from agentic_rag.knowledge_collection import (
    FetchResult,
    KnowledgeCollector,
    LibraryHoursSource,
    Scope,
    SocietiesSource,
    StudentEventsSource,
)

HWU = "https://www.hw.ac.uk"
SCOPE = Scope(
    include={"www.hw.ac.uk": ["/students", "/about/professional-services/information-services"]},
    exclude=["/profiles", "/news"],
)


def html_page(title: str, body: str = "", links: list[str] = ()) -> str:
    anchors = "".join(f'<a href="{href}">link</a>' for href in links)
    return (
        f"<html><head><title>{title} | Heriot-Watt University</title></head><body>"
        "<nav><a href='/'>Home</a> Menu Search</nav>"
        f"<main><h1>{title}</h1>{body}<p>{anchors}</p></main>"
        "<footer>Copyright Heriot-Watt University</footer></body></html>"
    )


def sitemap(*paths: str) -> str:
    return "<urlset>" + "".join(f"<url><loc>{HWU}{p}</loc></url>" for p in paths) + "</urlset>"


class FakeWeb:
    """Serves canned responses by URL and records what was fetched."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.fetched: list[str] = []

    def __call__(self, url: str) -> FetchResult:
        self.fetched.append(url)
        if url.endswith("/robots.txt"):
            return FetchResult(url=url, status=200, content_type="text/plain", text=self.pages.get(url, ""))
        if url not in self.pages:
            return FetchResult(url=url, status=404, content_type="text/html", text="")
        content_type = "application/xml" if url.endswith(".xml") else "text/html"
        return FetchResult(url=url, status=200, content_type=content_type, text=self.pages[url])


ENROLMENT_TEXT = (
    "<p>Online enrolment opens on 4 August for continuing students. You must complete it "
    "before registering for courses in the Student Portal.</p>"
)


def test_in_scope_sitemap_pages_are_collected_and_out_of_scope_pages_are_never_fetched():
    web = FakeWeb({
        f"{HWU}/sitemap.xml": sitemap("/students/enrolment", "/profiles/uk/jane-doe", "/news/2026/big-news"),
        f"{HWU}/students/enrolment": html_page("Enrolment", ENROLMENT_TEXT),
        f"{HWU}/profiles/uk/jane-doe": html_page("Jane Doe", "<p>Professor of things.</p>"),
        f"{HWU}/news/2026/big-news": html_page("Big news", "<p>News.</p>"),
    })

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(sitemaps=[f"{HWU}/sitemap.xml"])

    assert [(p["title"], p["url"]) for p in corpus["pages"]] == [("Enrolment", f"{HWU}/students/enrolment")]
    assert "Online enrolment opens on 4 August" in corpus["pages"][0]["extracted_text"]
    assert not any("/profiles/" in url or "/news/" in url for url in web.fetched)


LIBRARY_HOURS = "/about/professional-services/information-services/library-essentials/library-opening-hours"


def test_in_scope_pages_linked_from_collected_pages_are_found_even_when_missing_from_the_sitemap():
    web = FakeWeb({
        f"{HWU}/sitemap.xml": sitemap(LIBRARY_HOURS),
        f"{HWU}{LIBRARY_HOURS}": html_page(
            "Library opening hours",
            "<p>Choose your campus library to see its opening and staffed hours.</p>",
            links=[f"{LIBRARY_HOURS}/edinburgh", "/profiles/uk/librarian", "https://twitter.com/hwlibrary"],
        ),
        f"{HWU}{LIBRARY_HOURS}/edinburgh": html_page(
            "Edinburgh library and Learning Commons 1 & 2",
            "<p>Your library is open 24/7 during semester 1 and 2 and staffed 7 days a week.</p>",
        ),
    })

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(sitemaps=[f"{HWU}/sitemap.xml"])

    urls = [p["url"] for p in corpus["pages"]]
    assert f"{HWU}{LIBRARY_HOURS}/edinburgh" in urls
    assert not any("profiles" in u or "twitter" in u for u in web.fetched)


def test_pages_disallowed_by_robots_txt_are_not_fetched():
    web = FakeWeb({
        f"{HWU}/robots.txt": "User-agent: *\nDisallow: /students/private\n",
        f"{HWU}/sitemap.xml": sitemap("/students/enrolment", "/students/private/area"),
        f"{HWU}/students/enrolment": html_page("Enrolment", ENROLMENT_TEXT),
        f"{HWU}/students/private/area": html_page("Private", ENROLMENT_TEXT),
    })

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(sitemaps=[f"{HWU}/sitemap.xml"])

    assert [p["url"] for p in corpus["pages"]] == [f"{HWU}/students/enrolment"]
    assert f"{HWU}/students/private/area" not in web.fetched


def test_tables_are_kept_as_readable_rows_and_related_content_teasers_are_dropped():
    calendar = (
        "<p>Key dates for the 2026/27 academic year on UK campuses.</p>"
        "<table><tr><th>Period</th><th>Dates</th></tr>"
        "<tr><td>Semester 1 teaching</td><td>14 September to 4 December 2026</td></tr>"
        "<tr><td>Semester 1 exams</td><td>7 to 18 December 2026</td></tr></table>"
        "<h2>Related content</h2><p>Read our latest student blog posts.</p>"
    )
    web = FakeWeb({
        f"{HWU}/sitemap.xml": sitemap("/students/academic-calendar"),
        f"{HWU}/students/academic-calendar": html_page("Academic calendar", calendar),
    })

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(sitemaps=[f"{HWU}/sitemap.xml"])

    text = corpus["pages"][0]["extracted_text"]
    assert "Period: Semester 1 teaching | Dates: 14 September to 4 December 2026" in text
    assert "Period: Semester 1 exams | Dates: 7 to 18 December 2026" in text
    assert "latest student blog posts" not in text


def _day(date: str, rendered: str) -> dict:
    return {"date": date, "rendered": rendered}


LIBCAL_HOURS_API = "https://hw.ac.libcal.com/api_hours_grid.php?iid=854&format=json&weeks=2&systemTime=0"
LIBCAL_HOURS = {
    "locations": [
        {"lid": 635, "name": "Edinburgh Campus Library", "weeks": [{
            "Monday": _day("2026-11-02", "24 Hours"), "Saturday": _day("2026-11-07", "24 Hours"),
        }]},
        {"lid": 7584, "parent_lid": 635, "name": "Edinburgh Campus Library Staffed Hours", "weeks": [{
            "Monday": _day("2026-11-02", "8:30am - 8:30pm"), "Saturday": _day("2026-11-07", "9:45am - 5pm"),
        }]},
        {"lid": 2569, "name": "Borders Campus Library", "weeks": [{
            "Monday": _day("2026-11-02", "9am - 5pm"), "Saturday": _day("2026-11-07", "Closed"),
        }]},
    ]
}


def test_library_hours_from_libcal_become_one_cited_page_per_campus_library():
    import json

    web = FakeWeb({LIBCAL_HOURS_API: json.dumps(LIBCAL_HOURS)})
    collector = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0)

    corpus = collector.collect(
        library_hours=LibraryHoursSource(
            api_url=LIBCAL_HOURS_API,
            pages={635: f"{HWU}{LIBRARY_HOURS}/edinburgh"},
            default_page=f"{HWU}{LIBRARY_HOURS}",
        )
    )

    by_title = {p["title"]: p for p in corpus["pages"]}
    edinburgh = by_title["Edinburgh Campus Library opening hours"]
    assert edinburgh["url"] == f"{HWU}{LIBRARY_HOURS}/edinburgh"
    assert "Edinburgh Campus Library: Monday 2 November 2026: 24 Hours" in edinburgh["extracted_text"]
    assert "Edinburgh Campus Library Staffed Hours: Saturday 7 November 2026: 9:45am - 5pm" in edinburgh["extracted_text"]
    borders = by_title["Borders Campus Library opening hours"]
    assert borders["url"] == f"{HWU}{LIBRARY_HOURS}"
    assert "Saturday 7 November 2026: Closed" in borders["extracted_text"]


LIBCAL_EVENTS_API = "https://hw.ac.libcal.com/api_events.php?m=month&cid=9127&format=json"


def _event_table(title: str, link: str, start: str, end: str, location: str) -> str:
    row = '<tr class="{cls}"><td class="s-lc-ea-l">{label}:</td><td>{value}</td></tr>'
    return (
        '<table class="s-lc-ea-tb">'
        + row.format(cls="s-lc-ea-ttit", label="Title", value=f'<a href="{link}">{title}</a>')
        + row.format(cls="s-lc-ea-from", label="From", value=start)
        + row.format(cls="s-lc-ea-to", label="To", value=end)
        + row.format(cls="s-lc-ea-tloc", label="Location", value=location)
        + "</table>"
    )


def test_student_events_from_libcal_become_one_cited_page_per_event():
    feed = '<h3 class="s-lc-ea-h3">Events in the next month</h3>' + _event_table(
        "Yoga", "https://hw.ac.libcal.com/event/4613271?hs=a",
        "09:00 Monday, November 2, 2026", "10:00 Monday, November 2, 2026",
        "Jean Muir Student Village, Scottish Borders Campus",
    ) + _event_table(
        "Careers fair", "https://hw.ac.libcal.com/event/4700001",
        "11:00 Wednesday, November 11, 2026", "15:00 Wednesday, November 11, 2026",
        "James Watt Centre, Edinburgh Campus",
    )
    web = FakeWeb({LIBCAL_EVENTS_API: feed})

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(
        student_events=StudentEventsSource(api_url=LIBCAL_EVENTS_API)
    )

    by_title = {p["title"]: p for p in corpus["pages"]}
    careers = by_title["Careers fair (student event)"]
    assert careers["url"] == "https://hw.ac.libcal.com/event/4700001"
    assert "When: 11:00 Wednesday, November 11, 2026 to 15:00 Wednesday, November 11, 2026" in careers["extracted_text"]
    assert "Where: James Watt Centre, Edinburgh Campus" in careers["extracted_text"]
    assert by_title["Yoga (student event)"]["url"] == "https://hw.ac.libcal.com/event/4613271"


def test_each_page_is_collected_once_however_its_url_is_written_or_wherever_it_is_mirrored():
    enrolment = html_page(
        "Enrolment", ENROLMENT_TEXT, links=["/students/enrolment#apply", "/students/enrolment?utm_source=x"]
    )
    web = FakeWeb({
        f"{HWU}/sitemap.xml": sitemap("/students/enrolment/", "/students/enrol-now"),
        f"{HWU}/students/enrolment": enrolment,
        f"{HWU}/students/enrol-now": enrolment,  # same page published at a second address
    })

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(sitemaps=[f"{HWU}/sitemap.xml"])

    assert [p["url"] for p in corpus["pages"]] == [f"{HWU}/students/enrolment"]
    assert web.fetched.count(f"{HWU}/students/enrolment") == 1


def test_unreachable_pages_are_reported_and_collection_carries_on():
    class FlakyWeb(FakeWeb):
        def __call__(self, url):
            if url.endswith("/students/timeout"):
                self.fetched.append(url)
                raise TimeoutError("read timed out")
            return super().__call__(url)

    web = FlakyWeb({
        f"{HWU}/sitemap.xml": sitemap("/students/missing", "/students/timeout", "/students/enrolment"),
        f"{HWU}/students/enrolment": html_page("Enrolment", ENROLMENT_TEXT),
    })

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(sitemaps=[f"{HWU}/sitemap.xml"])

    assert [p["url"] for p in corpus["pages"]] == [f"{HWU}/students/enrolment"]
    assert {u["url"] for u in corpus["unavailable"]} == {f"{HWU}/students/missing", f"{HWU}/students/timeout"}


def test_each_host_is_paced_by_its_own_robots_crawl_delay():
    import json

    clock = {"now": 0.0}
    request_times: dict[str, list[float]] = {}

    class TimedWeb(FakeWeb):
        def __call__(self, url):
            request_times.setdefault(url, []).append(clock["now"])
            return super().__call__(url)

    events_api = "https://hw.ac.libcal.com/api_events.php?m=month&cid=9127"
    web = TimedWeb({
        "https://hw.ac.libcal.com/robots.txt": "User-agent: *\nCrawl-delay: 10\n",
        LIBCAL_HOURS_API: json.dumps(LIBCAL_HOURS),
        events_api: "",
    })
    collector = KnowledgeCollector(
        fetch=web, scope=SCOPE, delay_s=1.0,
        clock=lambda: clock["now"], sleep=lambda s: clock.__setitem__("now", clock["now"] + s),
    )

    collector.collect(
        library_hours=LibraryHoursSource(api_url=LIBCAL_HOURS_API, pages={}, default_page=f"{HWU}{LIBRARY_HOURS}"),
        student_events=StudentEventsSource(api_url=events_api),
    )

    assert request_times[events_api][0] - request_times[LIBCAL_HOURS_API][0] >= 10


UNION = "https://www.hwunion.com"


def test_content_on_sites_that_wrap_the_whole_page_in_a_form_is_still_extracted():
    union_scope = Scope(include={"www.hwunion.com": ["/your-union"]})
    page = (
        # Same shape as the real Students' Union site (an ASP.NET page): everything,
        # navigation included, sits inside one <form>, and there is no <main>.
        "<html><head><title>Eat / Drink / Shop</title></head><body id='msl'><form id='aspnetForm' action='/x'>"
        "<div class='header'><a href='/'>Home</a> About Us Get Involved Your Union</div>"
        "<div class='container'><div class='row'><div class='col-12'><h1>Eat / Drink / Shop</h1>"
        "<p>Whether you are looking for something to eat, a night out with friends or to pick up "
        "your Uni essentials on campus, our outlets are the places you want to be.</p>"
        "<input type='text' name='search'><button>Search</button></div></div></div>"
        "</form></body></html>"
    )
    web = FakeWeb({f"{UNION}/your-union/eat-drink-shop": page})

    corpus = KnowledgeCollector(fetch=web, scope=union_scope, delay_s=0).collect(
        seed_urls=[f"{UNION}/your-union/eat-drink-shop"]
    )

    assert "our outlets are the places you want to be" in corpus["pages"][0]["extracted_text"]


def test_http_links_to_a_page_are_treated_as_the_https_page():
    web = FakeWeb({
        f"{HWU}/sitemap.xml": sitemap("/students/enrolment"),
        f"{HWU}/students/enrolment": html_page("Enrolment", ENROLMENT_TEXT, links=["http://www.hw.ac.uk/students/enrolment"]),
    })

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(sitemaps=[f"{HWU}/sitemap.xml"])

    assert [p["url"] for p in corpus["pages"]] == [f"{HWU}/students/enrolment"]
    assert "http://www.hw.ac.uk/students/enrolment" not in web.fetched


SOCIETIES_FEED = "https://www.hwunion.com/svc/feeds/groups/6033?subtree=true&attributesets=Status"
SOCIETIES_PAGE = "https://www.hwunion.com/get-involved/societies"


def test_societies_from_the_union_feed_become_cited_pages_and_a_directory_by_category():
    import json

    feed = [
        {"Id": 6193, "Name": "Chinese Students and Scholars Association", "Type": "Society", "Category": "Cultural",
         "EmailAddress": "hwucssa@outlook.com", "InstagramUsername": "", "Url": "https://www.hwunion.com/societies/6193/",
         "LongDescription": None, "Description": None, "Attributes": [{"Set": "Status", "Values": ["Affiliated"]}]},
        {"Id": 7001, "Name": "Hillwalking Club", "Type": "Society", "Category": "Activity",
         "EmailAddress": "hillwalking@hw.ac.uk", "InstagramUsername": "hwhillwalking",
         "Url": "https://www.hwunion.com/societies/7001/",
         "LongDescription": "<p>We climb Munros most <strong>Saturdays</strong>. Beginners welcome!</p>"},
    ]
    web = FakeWeb({SOCIETIES_FEED: json.dumps(feed)})

    corpus = KnowledgeCollector(fetch=web, scope=SCOPE, delay_s=0).collect(
        societies=SocietiesSource(api_url=SOCIETIES_FEED, directory_page=SOCIETIES_PAGE)
    )

    by_title = {p["title"]: p for p in corpus["pages"]}
    hillwalking = by_title["Hillwalking Club (society)"]
    assert hillwalking["url"] == "https://www.hwunion.com/societies/7001"
    assert "We climb Munros most Saturdays. Beginners welcome!" in hillwalking["extracted_text"]
    assert "Contact: hillwalking@hw.ac.uk" in hillwalking["extracted_text"]
    assert "Instagram: @hwhillwalking" in hillwalking["extracted_text"]
    directory = by_title["Students' Union societies directory"]
    assert directory["url"] == SOCIETIES_PAGE
    assert "Activity: Hillwalking Club" in directory["extracted_text"]
    assert "Cultural: Chinese Students and Scholars Association" in directory["extracted_text"]
