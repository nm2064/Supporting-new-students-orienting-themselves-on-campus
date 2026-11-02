"""Read private service configuration, then ingest as the restricted app user."""
import json
import os
from pathlib import Path
import pwd
import sys

from dotenv import load_dotenv

load_dotenv('/etc/unibot.env', override=True)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', ANONYMIZED_TELEMETRY='False')
account = pwd.getpwnam('unibot')
os.setgroups([])
os.setgid(account.pw_gid)
os.setuid(account.pw_uid)
os.chdir('/opt/unibot/app')
sys.path.insert(0, '/opt/unibot/app')
from agentic_rag.rag import get_rag

rag = get_rag()
print(json.dumps({'embedding_dimensions': len(rag.backend.embed_texts(['University campus information'])[0])}), flush=True)
result = rag.ingest()
print(json.dumps({'status': result['status'], 'chunks': result['chunks'], 'collection_count': rag.vector_store.collection.count()}), flush=True)
