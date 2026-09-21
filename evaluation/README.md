# Evaluation

`results/baseline-2026-04-09.json` preserves the supplied evaluation output. Its date comes from the saved file timestamp. It contains 10 queries and reports response timing, confidence, retrieval, citations and keyword matches. These figures are historical output, not a new evaluation of the organised repository.

The runner sends real requests to a configured local backend and may incur external API usage:

```console
python evaluation/run_evaluation_tests.py
python evaluation/run_evaluation_tests.py --base-url http://127.0.0.1:8000 --output evaluation/results/latest.json
```

The default `results/latest.json` is ignored by Git so repeated runs do not overwrite the committed baseline. To retain another experiment, choose a distinct output name and review it before committing.

Keyword overlap and model confidence are limited indicators; they do not independently establish factual correctness or user-study outcomes. This runner is separate from the offline regression suite:

```console
python -m pytest agentic_rag/tests -q -p no:cacheprovider
```
