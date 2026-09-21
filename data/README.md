# Research data

These files preserve the university-information collection used during development. Source URLs and available crawl metadata are retained in the records. They are research snapshots, not a live directory of university services.

| Files | Role and date evidence |
| --- | --- |
| `raw/societies.jsonl`, `raw/societies.txt` | HW Union society records; saved 23 February 2026 |
| `raw/hw_useful_links.jsonl`, `raw/hw_useful_links.txt` | Website crawl outputs; saved 10 March 2026 |
| `processed/hw_student_links_curated.*` | Curated link collection; saved 10 March 2026 |
| `processed/hw_student_links_shortlist.*` | Shortlisted links; saved 10 March 2026 |
| `processed/hw_student_links_clean.*` | Cleaned link collection used by the extractor; saved 10 March 2026 |
| `processed/hw_student_pages_extracted.json` | Extracted page records; saved 10 March 2026 |

Dates above describe saved-file evidence, not independent verification of when every record was written. Human-readable and structured variants are retained for traceability. The filtering steps between curated, shortlisted and cleaned files have no separate saved script in the supplied snapshots; they are not claimed to be fully reproducible.

The backend knowledge artifacts remain under `agentic_rag/`. Do not replace them automatically with a root data file: compare the formats and set `KNOWLEDGE_FILE` deliberately. Extracting pages updates the processed JSON and backend knowledge text; it does not automatically replace the separately configured backend JSON corpus.

## Collection tools

Install `scripts/requirements.txt` in your environment, then run from the repository root:

```console
python scripts/crawl_hw_website.py --help
python scripts/scrape_hwunion_societies.py --help
python scripts/extract_hw_student_page_text.py --help
python scripts/scrape_text_from_html.py --help
```

Example collection commands:

```console
python scripts/crawl_hw_website.py --sites all --max-pages-per-site 150
python scripts/scrape_hwunion_societies.py
python scripts/extract_hw_student_page_text.py --input data/processed/hw_student_links_clean.json
```

Default paths are anchored to the repository, independent of the working directory. Crawls write to `data/raw/`; page extraction writes to `data/processed/` and the backend knowledge text file. Logs go to ignored `logs/`. Explicit relative path arguments are relative to the caller's working directory. To preserve existing crawl files, use the crawler's output options with a new destination before running it.

The websites and retained third-party documents keep their original ownership. Existing URLs, source information and acknowledgements have been preserved.
