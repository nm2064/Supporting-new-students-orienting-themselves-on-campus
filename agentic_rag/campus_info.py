"""Student-facing information from the active, cited knowledge corpus."""
import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from .settings import get_settings

class CampusInfo:
    def __init__(self, corpus: Path, now=datetime.now):
        self.corpus, self.now = corpus, now

    def pages(self):
        try:
            return json.loads(self.corpus.read_text(encoding="utf-8"))["pages"]
        except (OSError, ValueError, KeyError):
            raise HTTPException(503, "Campus information is temporarily unavailable")

    def events(self, campus=None):
        events = []
        for page in self.pages():
            event = page.get("event")
            if not event or (campus and event.get("campus") != campus):
                continue
            try:
                end = datetime.fromisoformat(event.get("end") or event["start"])
                now = self.now()
                if end.tzinfo:
                    now = now.astimezone(end.tzinfo)
                if end >= now:
                    events.append(event)
            except (ValueError, TypeError, KeyError):
                continue
        return sorted(events, key=lambda event: event["start"])

    def services(self, campus=None):
        terms = ("support", "wellbeing", "careers", "accommodation", "it services", "registry", "visa", "disability")
        results = []
        seen = set()
        for page in self.pages():
            title, url = page.get("title", ""), page.get("url", "")
            if not url.startswith("https://") or url in seen or not any(term in title.lower() for term in terms):
                continue
            text = page.get("extracted_text", "")
            if campus and campus.replace("-", " ") not in (title + " " + text).lower():
                continue
            seen.add(url)
            results.append({"title": title, "url": url, "description": text[:350]})
        return sorted(results, key=lambda item: item["title"])

def get_campus_info():
    return CampusInfo(get_settings().knowledge_file)

router = APIRouter(prefix="/api", tags=["campus information"])

@router.get("/events")
def events(campus: str | None = None, info: CampusInfo = Depends(get_campus_info)):
    return info.events(campus)

@router.get("/services")
def services(campus: str | None = None, info: CampusInfo = Depends(get_campus_info)):
    return info.services(campus)

@router.get("/opening-hours")
def opening_hours(campus: str | None = None, info: CampusInfo = Depends(get_campus_info)):
    return [{"title": p["title"], "url": p["url"], "description": p.get("extracted_text", "")}
            for p in info.pages()
            if "opening" in p.get("title", "").lower() and "hour" in p.get("title", "").lower()
            and p.get("url", "").startswith("https://")
            and (not campus or campus.replace("-", " ") in (p.get("title", "") + " " + p.get("extracted_text", "")).lower())]
