"""Privately merge the supplied Azure connection into the service environment."""
import os
from pathlib import Path
import shutil
import time

upload = Path('/home/ubuntu/unibot-configuration-upload.env')
try:
    config = dict(line.split('=', 1) for line in upload.read_text().splitlines() if '=' in line and not line.startswith('#'))
    required = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_CHAT_DEPLOYMENT', 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT']
    if any(not config.get(k, '').strip() for k in required):
        raise SystemExit('Incomplete Azure configuration')
    updates = {k: config[k] for k in required}
    updates.update(USE_AZURE_OPENAI='true', CHAT_MODEL='gpt-6-luna', TRANSLATION_MODEL='gpt-6-luna',
                   CHAT_MAX_TOKENS='1600', CHUNK_SIZE='1200', CHUNK_OVERLAP='150', TOP_K='12')
    env = Path('/etc/unibot.env')
    shutil.copy2(env, env.with_name('unibot.env.before-azure-'+str(int(time.time()))))
    existing = env.read_text().splitlines()
    existing = [line for line in existing if line.partition('=')[0] not in updates]
    existing += [k+'='+v for k, v in updates.items()]
    candidate = env.with_name('unibot.env.new')
    fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write('\n'.join(existing)+'\n')
    os.replace(candidate, env)
    print('Azure configuration saved privately for gpt-6-luna and text-embedding-3-small.')
finally:
    upload.unlink(missing_ok=True)
