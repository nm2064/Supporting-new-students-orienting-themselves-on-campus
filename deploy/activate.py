"""Install the isolated UniBot service and append its virtual host safely."""
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

if os.geteuid() != 0:
    raise SystemExit("Run with sudo")
hostname = "uni-bot.duckdns.org"
source = Path("/opt/unibot/app")
env = Path("/etc/unibot.env")
if not env.exists():
    config = "\n".join([
        "USE_AZURE_OPENAI=false", "OPENAI_API_KEY=", "ORS_API_KEY=",
        "CHAT_MODEL=gpt-4o-mini", "EMBEDDING_MODEL=text-embedding-3-small",
        "PUBLIC_DEPLOYMENT=true", f"PUBLIC_ORIGIN=https://{hostname}",
        "PUBLIC_SESSION_SECRET="+secrets.token_hex(32),
        "PUBLIC_DAILY_REQUEST_LIMIT=200", "PUBLIC_LIMIT_DB=/var/lib/unibot/limits.sqlite3",
        "CHROMA_PERSIST_DIR=/var/lib/unibot/chroma",
        "KNOWLEDGE_FILE=/opt/unibot/app/agentic_rag/knowledge_extracted_pages.json",
        "MAP_ENABLE_TRANSLATION_FALLBACK=false", "LOG_LEVEL=WARNING", "",
    ])
    fd = os.open(env, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(config)
shutil.copyfile(source / "deploy/unibot.service", "/etc/systemd/system/unibot.service")
subprocess.run(["systemctl", "daemon-reload"], check=True)
subprocess.run(["systemctl", "enable", "--now", "unibot"], check=True)
for _ in range(30):
    result = subprocess.run(["curl", "--silent", "--fail", "http://127.0.0.1:8001/"], stdout=subprocess.DEVNULL)
    if result.returncode == 0:
        break
    time.sleep(1)
else:
    raise SystemExit("UniBot has not started; Caddy unchanged")
caddy = Path("/etc/caddy/Caddyfile")
original = caddy.read_text()
block = """

uni-bot.duckdns.org {
    encode gzip
    request_body {
        max_size 16KB
    }
    reverse_proxy 127.0.0.1:8001 {
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
        transport http {
            dial_timeout 5s
            read_timeout 120s
        }
    }
    header {
        Strict-Transport-Security "max-age=31536000"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "strict-origin-when-cross-origin"
        -Server
    }
}
"""
if hostname not in original:
    candidate = caddy.with_name("Caddyfile.unibot-candidate")
    candidate.write_text(original+block)
    subprocess.run(["caddy", "validate", "--config", str(candidate), "--adapter", "caddyfile"], check=True)
    backup = caddy.with_name("Caddyfile.before-unibot-"+str(int(time.time())))
    shutil.copy2(caddy, backup)
    shutil.copyfile(candidate, caddy)
    reload_result = subprocess.run(["systemctl", "reload", "caddy"])
    if reload_result.returncode:
        shutil.copyfile(backup, caddy)
        subprocess.run(["systemctl", "reload", "caddy"], check=True)
        raise SystemExit("Caddy reload failed; restored previous configuration")
print("UniBot service started and virtual host installed. AI credentials still need configuration.")
