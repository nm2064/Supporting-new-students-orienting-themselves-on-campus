# Development history

## Source evidence and method

This Git history was reconstructed on 21 September 2026 from saved local project versions. The application work predates its upload. The reconstructed commits group real differences between those saved versions; they are not a recovered record of every original edit or commit.

Both the author and committer are `nm2064 <nomanmaqsudi@outlook.com>`. Historical commits use the latest relevant saved-file modification timestamp in their group, to the nearest whole second, with the local UK offset. These timestamps are estimates from file evidence rather than independently verified completion times. New organisation and documentation changes use their actual creation date.

| Snapshot | Evidence used |
| --- | --- |
| `UniBot 27.02` | Earlier chatbot, retrieval experiments and society collection; files saved through 25 February 2026 |
| `UniBot 07.03.zip` | Navigation changes; ZIP entries extend through 8 March 2026 despite the archive filename |
| `UniBot 27.07 (Dis)` | Later backend, knowledge collection, April evaluation and presentation; saved contents extend through 17 April |
| Current `UniBot` folder | July launcher/frontend serving updates and August ignore rules |

The [original Gantt chart](../Gantt%20Chart/README.md) provides planned milestones, not proof of implementation dates. Its existing GitHub history has been preserved. The original local initial commit (`1923ce69e7759c98a5be15f7849fcaecd5109c3f`) is retained locally as a backup and is not substituted for the existing GitHub history.

## Reconstructed milestones

| Saved date | Commit | Change |
| --- | --- | --- |
| 2026-02-03 | `ca128f14` | chore(research): preserve retrieval experiments and HTML text extraction |
| 2026-02-23 | `0fc0a7d3` | feat(data): collect Heriot-Watt Union society information |
| 2026-02-25 | `75bdc1b5` | feat(chat): add multilingual RAG assistant and web interface |
| 2026-03-08 | `905a7e29` | feat(maps): add campus places and walking route integration |
| 2026-03-09 | `fe075934` | data(campus): update campus place records |
| 2026-03-10 | `bf9945b2` | chore(archive): separate legacy retrieval experiments from the application |
| 2026-03-10 | `f82a2924` | feat(data): collect and prepare university knowledge sources |
| 2026-04-09 | `919b6516` | refactor(backend): separate API services and integrate student knowledge |
| 2026-04-09 | `be41b995` | test(evaluation): record campus and multilingual query results |
| 2026-04-17 | `052c1162` | docs: add project presentation and interface screenshots |
| 2026-07-27 | `5e720571` | feat(dev): add a single-command local application launcher |
| 2026-08-19 | `869df249` | chore(git): exclude credentials and local runtime artifacts |

Some March backend files depend on a RAG implementation whose next surviving version is from April; those changes are grouped with that April snapshot rather than inventing an unavailable March version. Snapshot boundaries are preserved by content comparison, while individual import commits are not asserted to be independently deployable.

## Publication preparation

The September organisation moves tools into `scripts/`, collection artifacts into `data/`, evaluation materials into `evaluation/`, and guides and presentation into `docs/`. Runtime backend knowledge and frontend entry points retain their locations. File references and default output paths have been updated.

Historical hard-coded credentials were removed before committing imported versions. Notebook execution outputs and counts were cleared in all imported snapshots. These publication changes are explicit exceptions to byte-for-byte preservation; source cells, reference materials and acknowledgements remain. Local credentials, environments, caches and vector stores are excluded.

The final layout passed all 16 existing backend regression tests, command-line help checks from outside the repository, static file/API-documentation serving checks, JSON/JSONL parsing and local documentation-link checks. These checks use no live model or route-provider calls. The saved April evaluation was preserved without rerunning or changing its results.
