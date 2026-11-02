"""Controls for the single-worker public demo deployment."""
import os
import sqlite3
import time
import uuid
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def public_mode():
    return os.getenv("PUBLIC_DEPLOYMENT", "false").lower() == "true"


class PublicAccessMiddleware(BaseHTTPMiddleware):
    @staticmethod
    def _consume_request_allowance(request):
        """Count one AI request against the per-IP and daily limits.

        Returns a 429 response when a limit is reached, otherwise None.
        Counters persist in SQLite, so restarts do not reset them.
        """
        now = int(time.time())
        dbpath = Path(os.getenv("PUBLIC_LIMIT_DB", "/var/lib/unibot/limits.sqlite3"))
        dbpath.parent.mkdir(parents=True, exist_ok=True)
        ip = request.client.host if request.client else "unknown"
        with sqlite3.connect(dbpath, timeout=5) as db:
            db.execute("CREATE TABLE IF NOT EXISTS limits (bucket TEXT PRIMARY KEY, n INTEGER NOT NULL, expiry INTEGER NOT NULL)")
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM limits WHERE expiry < ?", (now,))
            buckets = [(f"ip:{ip}:{now//60}", 10, (now//60+1)*60),
                       (f"day:{now//86400}", int(os.getenv("PUBLIC_DAILY_REQUEST_LIMIT", "200")), (now//86400+1)*86400)]
            for key, limit, expiry in buckets:
                row = db.execute("SELECT n FROM limits WHERE bucket=?", (key,)).fetchone()
                if row and row[0] >= limit:
                    return JSONResponse({"detail": "Demo request limit reached. Please try again later."}, status_code=429, headers={"Retry-After": str(expiry-now)})
            for key, _, expiry in buckets:
                db.execute("INSERT INTO limits VALUES (?,1,?) ON CONFLICT(bucket) DO UPDATE SET n=n+1", (key, expiry))
        return None

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in {"/ingest", "/docs", "/redoc", "/openapi.json", "/health"} or path.startswith("/session/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        if request.method not in {"GET", "HEAD", "POST"}:
            return JSONResponse({"detail": "Method not allowed"}, status_code=405)
        if request.method == "POST":
            if path == "/rag-chat":
                from .settings import get_settings, ConfigurationError
                try:
                    get_settings().validate_required()
                except ConfigurationError:
                    return JSONResponse({"detail": "The AI assistant is being configured. Please check back soon."}, status_code=503)
            origin = request.headers.get("origin")
            expected = os.environ.get("PUBLIC_ORIGIN", "")
            if origin and origin != expected:
                return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
            if path not in {"/rag-chat", "/api/route", "/reset-chat"}:
                return JSONResponse({"detail": "Not found"}, status_code=404)
            # Caddy also enforces this limit, including chunked requests.
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
            if length > 16384:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
            # The limits protect AI quota. Starting a new conversation (sent on
            # every page load) costs no AI calls, so it does not count.
            if path != "/reset-chat":
                limited = self._consume_request_allowance(request)
                if limited is not None:
                    return limited
            # Session IDs come from a signed, HTTP-only browser cookie, not user input.
            import hashlib
            import hmac
            secret = os.environ["PUBLIC_SESSION_SECRET"].encode()
            cookie = request.cookies.get("unibot_session", "")
            session, _, signature = cookie.partition(".")
            valid = len(session) == 32 and hmac.compare_digest(signature, hmac.new(secret, session.encode(), hashlib.sha256).hexdigest())
            if not valid:
                session = uuid.uuid4().hex
            if path == "/reset-chat":
                from . import rag
                if rag.rag_instance is not None:
                    rag.rag_instance.clear_session(session)
                session = uuid.uuid4().hex
            signature = hmac.new(secret, session.encode(), hashlib.sha256).hexdigest()
            request.state.public_session = session
            response = JSONResponse({"status": "reset"}) if path == "/reset-chat" else await call_next(request)
            response.set_cookie("unibot_session", session+"."+signature, secure=True, httponly=True, samesite="lax", max_age=86400)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
