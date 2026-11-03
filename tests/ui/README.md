# Frontend browser tests

From the repository root:

    python -m pip install -r tests/ui/requirements.txt
    python -m playwright install chromium
    python -m pytest tests/ui -q

Tests start a temporary local static server and Chromium. API fixtures prevent real AI requests. CDN access is needed for the frontend's existing React, Babel, Tailwind and Leaflet dependencies.

The tests exercise rendered public pages: home photography/actions, mobile map space, event source links, settings persistence/Arabic RTL at several viewport sizes, and chat question starters. Backend contracts remain covered by agentic_rag/tests.
