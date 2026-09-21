# Frontend integration

The existing frontend is `index.html`, served at `/` by `agentic_rag/app.py`. Keep it on the same origin as the API when running locally. The integration is already implemented; no replacement component is required.

## Chat contract

Send JSON to `POST /rag-chat`, for example:

```json
{
  "message": "Where is the library?",
  "conversation_id": "example-session",
  "ui_language": "en",
  "auto_detect": true,
  "campus": "Edinburgh"
}
```

`location_context` is optional. Older clients may use `question` and `session_id`. See `agentic_rag/schemas.py` and the running `/docs` page for field definitions.

The response contains `answer_markdown`, `effective_language`, `citations`, `follow_up_suggestions`, `needs_human_handoff`, `confidence` and `conversation_id`. Compatibility fields include `answer`, `sources`, `used_retrieval`, `clarifying_question` and `session_id`.

Retain the conversation identifier for follow-up messages, render source links from the response, and handle unsuccessful HTTP responses visibly. The current API returns complete responses; it does not expose a streaming chat endpoint.

## Other endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Configuration, knowledge and vector-store status |
| `POST /ingest` | Ingest the configured knowledge file; body `{"force": false}` |
| `GET /session/{id}` | Retrieve conversation history |
| `DELETE /session/{id}` | Clear a conversation |
| `GET /api/places` | Search campus places |
| `GET /api/places/{id}` | Look up a place |
| `POST /api/route` | Request a walking route |

Map payloads are defined in `agentic_rag/maps/models.py`. Route requests require configured OpenRouteService access.

## Local checks

Start the application, open `/`, send a greeting and a university-information question, follow a cited source, then test a follow-up and a campus route. A missing external key may prevent live answers or directions even when the page loads correctly. Backend regression tests cover API contracts using substitute services.
