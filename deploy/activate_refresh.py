"""Switch to a validated refresh with a rollback copy of service settings."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

snapshot = Path(sys.argv[1]).resolve()
base = Path('/var/lib/unibot/refreshes').resolve()
if snapshot.parent != base:
    raise SystemExit('Snapshot is outside the refresh directory')
manifest = json.loads((snapshot / 'manifest.json').read_text())
if manifest['stage'] != 'ready':
    raise SystemExit('Refresh has not passed validation')
env = Path('/etc/unibot.env')
backup = env.with_name('unibot.env.before-refresh-'+manifest['run_id'])
shutil.copy2(env, backup)
changes = {'KNOWLEDGE_FILE': str(snapshot / 'knowledge.json'), 'CHROMA_PERSIST_DIR': str(snapshot / 'chroma')}
lines = [line for line in env.read_text().splitlines() if line.partition('=')[0] not in changes]
lines.extend(key+'='+value for key,value in changes.items())
candidate = env.with_name('unibot.env.refresh-candidate')
fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(fd, 'w') as f:
    f.write('\n'.join(lines)+'\n')
os.replace(candidate, env)
subprocess.run(['systemctl', 'restart', 'unibot'], check=True)
for _ in range(30):
    health = subprocess.run(['curl', '-fsS', 'http://127.0.0.1:8001/'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if health.returncode == 0:
        break
    time.sleep(1)
else:
    shutil.copy2(backup, env)
    subprocess.run(['systemctl', 'restart', 'unibot'], check=True)
    raise SystemExit('New service did not start; restored previous settings')
manifest.update(stage='active', activated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'stage':'active','pages':manifest['fresh_pages'],'chunks':manifest['chunks'],'run_id':manifest['run_id']}))
