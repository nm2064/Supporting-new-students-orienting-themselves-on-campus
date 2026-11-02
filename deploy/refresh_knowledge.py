"""Collect a fresh knowledge corpus and build a separate, validated RAG index.

Run as root to load the service configuration; collection and indexing then run
under the unprivileged unibot account. Activation is a separate root operation
(deploy/activate_refresh.py), so a bad refresh never replaces the live index.

Collection is done by agentic_rag.knowledge_collection: sitemaps plus in-scope
link following on the university and partner sites, and LibCal for library
opening hours and student events.
"""
import hashlib
import json
import os
import pwd
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

APP = Path('/opt/unibot/app')
load_dotenv('/etc/unibot.env', override=True)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', ANONYMIZED_TELEMETRY='False')
account = pwd.getpwnam('unibot')
os.setgroups([])
os.setgid(account.pw_gid)
os.setuid(account.pw_uid)
os.chdir(APP)
sys.path[:0] = [str(APP)]
from agentic_rag import knowledge_collection as kc  # noqa: E402
from agentic_rag.rag import AgenticRAG  # noqa: E402
from agentic_rag.settings import get_settings  # noqa: E402

# Pages a good corpus must contain; a refresh missing them is not activated.
REQUIRED_URLS = [f'{kc.LIBRARY_HOURS_PAGE}/edinburgh']
VALIDATION_QUESTIONS = [
    ('When is the Edinburgh campus library open?', f'{kc.LIBRARY_HOURS_PAGE}/edinburgh'),
    ('How do I enrol as a continuing student?', None),
    ('Where can a student find accommodation support?', None),
]

settings = get_settings()
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
snapshot = Path('/var/lib/unibot/refreshes') / run_id
snapshot.mkdir(parents=True, exist_ok=False)
manifest_path = snapshot / 'manifest.json'
manifest = {'run_id': run_id, 'stage': 'collecting', 'started_at': datetime.now(timezone.utc).isoformat()}


def save_manifest():
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def progress(**values):
    print(json.dumps(values, ensure_ascii=True), flush=True)


def fail(reason):
    manifest.update(stage='failed', reason=reason)
    save_manifest()
    raise SystemExit(reason)


save_manifest()
previous = json.loads(settings.knowledge_file.read_text(encoding='utf-8')).get('pages', [])
manifest.update(previous_pages=len(previous))

collector = kc.KnowledgeCollector(
    fetch=kc.http_fetch(kc.DEFAULT_USER_AGENT),
    scope=kc.PRODUCTION_SCOPE,
    delay_s=1.0,
    site_labels=kc.PRODUCTION_SITE_LABELS,
)
corpus = collector.collect(
    sitemaps=kc.PRODUCTION_SITEMAPS,
    seed_urls=kc.PRODUCTION_SEEDS,
    library_hours=kc.PRODUCTION_LIBRARY_HOURS,
    student_events=kc.PRODUCTION_STUDENT_EVENTS,
)
pages, unavailable = corpus['pages'], corpus['unavailable']
corpus_path = snapshot / 'knowledge.json'
corpus_path.write_text(json.dumps({
    'total_pages': len(pages), 'refreshed_at': datetime.now(timezone.utc).isoformat(), 'pages': pages,
}, ensure_ascii=False, indent=1), encoding='utf-8')
(snapshot / 'unavailable-pages.json').write_text(json.dumps(unavailable, indent=1), encoding='utf-8')
manifest.update(fresh_pages=len(pages), unavailable=len(unavailable))
save_manifest()
progress(stage='collected', fresh_pages=len(pages), unavailable=len(unavailable))

if len(pages) < max(10, int(len(previous) * .9)):
    fail('Too few freshly collected pages to replace the current corpus')
missing = [url for url in REQUIRED_URLS if url not in {page['url'] for page in pages}]
if missing:
    fail(f'Required pages missing from the corpus: {missing}')

manifest.update(stage='indexing', knowledge_file=str(corpus_path), chroma_persist_dir=str(snapshot / 'chroma'))
save_manifest()
candidate = replace(settings, knowledge_file=corpus_path, chroma_persist_dir=snapshot / 'chroma',
                    ingest_state_file=snapshot / 'chroma/ingest_state.json')
rag = AgenticRAG.from_settings(candidate)
result = rag.ingest(force=True)
count = rag.vector_store.collection.count()
if not count or count != result['chunks']:
    fail('Candidate index count mismatch')

checks = []
for question, must_cite in VALIDATION_QUESTIONS:
    response = rag.chat(question, ui_language='en', auto_detect=False)
    cited = [citation['url'] for citation in response.citations]
    passed = bool(response.answer_markdown and response.used_retrieval and cited)
    if must_cite:
        passed = passed and must_cite in cited
    checks.append({'question': question, 'passed': passed, 'answer': response.answer_markdown, 'cited': cited})
(snapshot / 'validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
if not all(check['passed'] for check in checks):
    fail('Candidate did not pass live retrieval checks; see validation.json')

manifest.update(stage='ready', chunks=count, completed_at=datetime.now(timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest())
save_manifest()
progress(**manifest)
