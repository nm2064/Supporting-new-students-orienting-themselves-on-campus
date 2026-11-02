"""Refresh the curated public pages and build a separate, validated RAG index.

Run as root to load the service configuration; all collection and indexing then
runs under the existing unibot account. Activation is a separate root operation.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import pwd
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


APP = Path('/opt/unibot/app')
UA = 'UniBotPageExtractor/1.1 (+https://uni-bot.duckdns.org; campus-information-refresh)'
load_dotenv('/etc/unibot.env', override=True)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', ANONYMIZED_TELEMETRY='False')
account = pwd.getpwnam('unibot')
os.setgroups([])
os.setgid(account.pw_gid)
os.setuid(account.pw_uid)
os.chdir(APP)
sys.path[:0] = [str(APP), str(APP / 'scripts')]
from extract_hw_student_page_text import extract_page_text
from agentic_rag.settings import get_settings
from agentic_rag.rag import AgenticRAG

settings = get_settings()
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
snapshot = Path('/var/lib/unibot/refreshes') / run_id
snapshot.mkdir(parents=True, exist_ok=False)
manifest_path = snapshot / 'manifest.json'
manifest = {'run_id': run_id, 'stage': 'scraping', 'started_at': datetime.now(timezone.utc).isoformat()}

def save_manifest():
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def progress(**values):
    print(json.dumps(values, ensure_ascii=True), flush=True)

save_manifest()
old = json.loads(settings.knowledge_file.read_text(encoding='utf-8'))
previous = old.get('pages', [])
items = json.loads((APP / 'data/processed/hw_student_links_clean.json').read_text(encoding='utf-8')).get('links', [])
by_url = {item['url']: item for item in items}
for page in previous:
    by_url.setdefault(page['url'], page)
items = list(by_url.values())
manifest.update(previous_pages=len(previous), total_links=len(items))
save_manifest()
session = requests.Session()
session.headers['User-Agent'] = UA
session.max_redirects = 5
session.mount('https://', HTTPAdapter(max_retries=Retry(total=1, backoff_factor=.5, status_forcelist=[502, 503, 504])))
policies = {}
last_request = {}
records = []
failures = []
seen_urls = set()

def allowed_host(host):
    return any(host == name or host.endswith('.'+name) for name in ('hw.ac.uk', 'wattliving.co.uk', 'hwunion.com'))

def policy_for(url):
    parsed = urlparse(url)
    base = parsed.scheme+'://'+parsed.netloc
    if base not in policies:
        parser = RobotFileParser(base+'/robots.txt')
        try:
            response = session.get(base+'/robots.txt', timeout=(5, 12))
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                policies[base] = parser
            elif response.status_code == 404:
                policies[base] = None
            else:
                policies[base] = False
        except requests.RequestException:
            policies[base] = False
    return base, policies[base]

for index, item in enumerate(items, 1):
    url = item['url']
    try:
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not allowed_host(parsed.hostname or ''):
            raise ValueError('outside approved public sources')
        if item.get('is_document'):
            raise ValueError('non-HTML document')
        base, policy = policy_for(url)
        if policy is False or (policy and not policy.can_fetch(UA, url)):
            raise ValueError('robots policy prevents fetching')
        delay = max(.5, float(policy.crawl_delay(UA) or 0) if policy else 0)
        time.sleep(max(0, delay-(time.monotonic()-last_request.get(base, 0))))
        last_request[base] = time.monotonic()
        response = session.get(url, timeout=(5, 12))
        response.raise_for_status()
        if not allowed_host(urlparse(response.url).hostname or ''):
            raise ValueError('redirect outside approved sources')
        if 'html' not in response.headers.get('Content-Type', '').lower():
            raise ValueError('non-HTML response')
        text, headings, title = extract_page_text(response.text)
        if len(text) < 120 or 'page not found' in title.lower() or title.startswith('404'):
            raise ValueError('missing or low-text page')
        canonical = response.url.split('#')[0].rstrip('/')
        if canonical in seen_urls:
            raise ValueError('duplicate redirect destination')
        seen_urls.add(canonical)
        records.append({
            'category': item.get('category', 'campus_and_services'),
            'site_key': item.get('site_key', 'hwu'),
            'site_label': item.get('site_label', 'Heriot-Watt University'),
            'title': title or item.get('title') or url,
            'url': response.url, 'source_url': url,
            'summary': '', 'headings': headings, 'extracted_text': text,
            'extracted_at': datetime.now(timezone.utc).isoformat(),
        })
    except (requests.RequestException, ValueError) as error:
        failures.append({'url': url, 'reason': str(error)[:300]})
    if index % 10 == 0 or index == len(items):
        manifest.update(processed=index, fresh_pages=len(records), unavailable=len(failures))
        save_manifest()
        progress(stage='scraping', processed=index, total=len(items), fresh_pages=len(records), unavailable=len(failures))

corpus = snapshot / 'knowledge.json'
records.sort(key=lambda item: (item['category'], item['url']))
corpus.write_text(json.dumps({'total_pages': len(records), 'refreshed_at': datetime.now(timezone.utc).isoformat(), 'pages': records}, ensure_ascii=False, indent=2), encoding='utf-8')
(snapshot / 'unavailable-pages.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
if len(records) < max(10, int(len(previous)*.7)):
    manifest.update(stage='failed', reason='Too few freshly extracted pages to replace the current corpus')
    save_manifest()
    raise SystemExit(manifest['reason'])

manifest.update(stage='indexing', knowledge_file=str(corpus), chroma_persist_dir=str(snapshot / 'chroma'))
save_manifest()
progress(stage='indexing', fresh_pages=len(records), unavailable=len(failures))
candidate_settings = replace(settings, knowledge_file=corpus, chroma_persist_dir=snapshot / 'chroma', ingest_state_file=snapshot / 'chroma/ingest_state.json')
rag = AgenticRAG.from_settings(candidate_settings)
result = rag.ingest(force=True)
count = rag.vector_store.collection.count()
if not count or count != result['chunks']:
    raise RuntimeError('Candidate index count mismatch')
checks = []
for query in ['How can I contact the Edinburgh campus library?', 'Where can a student find accommodation support?']:
    response = rag.chat(query, ui_language='en', auto_detect=False)
    valid = bool(response.answer_markdown and response.used_retrieval and response.citations)
    checks.append({'query': query, 'passed': valid, 'answer': response.answer_markdown, 'citations': response.citations})
if not all(c['passed'] for c in checks):
    raise RuntimeError('Candidate did not pass live retrieval checks')
(snapshot / 'validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
manifest.update(stage='ready', chunks=count, completed_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest())
save_manifest()
progress(**manifest)
