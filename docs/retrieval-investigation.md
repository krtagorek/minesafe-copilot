# Retrieval investigation

## Scope and evidence

This investigation examined saved MineSafe retrieval data, corrected index initialization and reciprocal rank fusion (RRF), and created a separate comparison index. It does not establish improved answer quality or a production fix. The app still uses the original index at `data/processed/chroma_db`.

Fix commit: `fadec0cb68b130573398e01bb0084186726bdf15` — `Fix index duplication and reciprocal rank fusion identity`.

## Saved-data audit

The saved chunk cache contained 923 records with 923 unique full `page_content` values. The original Chroma collection contained 3,692 records with distinct record IDs but only 923 unique full texts: exactly four copies of every saved chunk, or 2,769 duplicate occurrences beyond the first. The text sets matched exactly.

Inspection of a temporary snapshot confirmed matching source/page identity and full metadata within every duplicate group. All embeddings were finite and had 1,536 dimensions. Embeddings were identical across all copies in 311 groups and differed in 612 groups. The largest observed coordinate difference was approximately 0.0188751; the minimum within-group pairwise cosine similarity was approximately 0.995714. The cause of these embedding differences was not established.

The original initialization called `Chroma.from_documents()` without explicit IDs. The installed integration generated fresh UUIDs and inserted documents into the existing collection. Reexecuting initialization could therefore append another corpus copy. Streamlit resource caching normally avoids initialization on ordinary reruns, but server restarts or cache invalidation can execute it again. Four insertion passes are consistent with the observed contents and timestamp groups; the audit did not prove which launches or commands performed them.

The previous RRF implementation used the first 100 text characters as identity and scored every returned occurrence. Repeated vector hits could accumulate multiple contributions within one retriever. Three prefix-collision groups involving seven distinct saved texts could also be merged incorrectly.

## Code fix

- Normal startup loads saved chunks and opens the existing collection without inserting documents. Missing saved data produces an explicit build instruction.
- Explicit index creation uses deterministic SHA-256 IDs derived from full text plus `source`, `source_file`, `page`, and `page_label`. Repeating a build of the same corpus upserts the same IDs.
- RRF uses the same full identity and counts each document at most once per retriever, while combining contributions across retrievers. Original result positions determine rank contributions.
- `scripts/compare_retrieval.py` uses the already-installed `langchain_chroma.Chroma` integration, accepts `--index-directory`, and queries only a temporary copy of the selected index. The directory option was added after the fix commit.

The fix does not delete legacy duplicate records or replace the app's index.

## Offline tests

All five tests in `tests/test_retrieval.py` passed using temporary indexes and deterministic local fake embeddings:

1. Repeated builds preserve record count and IDs; subsequent startup performs no ingestion or document embedding.
2. A missing index raises a build instruction without creating index files or constructing an embedding client.
3. Saved chunks load without ingestion; app initialization calls the saved-chunk loader.
4. Repeated hits within one retriever do not overpower cross-retriever agreement through duplicate contributions.
5. Different full texts sharing the first 100 characters, and identical texts with different source/page identity, remain distinct.

Command: `.venv/bin/python -B -m unittest discover -s tests -v`.

## Original-versus-clean comparison

The separate verified comparison index is at `data/processed/chroma_comparison_clean_v2`. It reused selected stored embeddings from the audited snapshot; it did not generate new embeddings. For each committed `document_id()` identity, the record with the lexicographically smallest original ID supplied the text, full metadata, and embedding. `selection_manifest.json` maps deterministic IDs to selected original IDs.

Offline verification confirmed exactly 923 records, matching identities, text, full metadata, exact selected vectors, and the manifest. Automatic embedding was disabled. The resolved schema and HNSW settings were preserved, including `l2` distance, `ef_construction=100`, `ef_search=100`, `max_neighbors=16`, `resize_factor=1.2`, `sync_threshold=1000`, `batch_size=100`, and `num_threads=10`.

The first creation attempt was rejected because installed Chroma does not accept schema and configuration simultaneously. Its directory, `data/processed/chroma_comparison_clean`, remains; the verified index uses `_v2` and was created with the supported schema API. No existing directory was overwritten.

The reported comparison observations are supplied by the user; the comparison outputs and exact query were not available for independent inspection in this investigation:

- Both comparison runs used corrected RRF.
- For one query, five distinct vector excerpts replaced the repeated vector results seen with the original index.

These observations concern retrieval diversity for one query. Selecting among differing stored vectors and rebuilding an approximate-neighbor graph can also affect rankings. They do not demonstrate improved answers, broader retrieval accuracy, or a production fix.

## Answer-bearing chunk and remaining uncertainty

Local inspection of the full saved chunks found a chunk containing both the ASTM F496-97 reference and a glove-testing interval:

> For example, high-voltage gloves should be tested every six months in accordance with ASTM F496-97

Whitespace is normalized from the saved chunk. Source: `PH26-V-1 Electrical Inspection Procedures Handbook 8_12_2026.pdf`; stored `page=6` (zero-based, seventh PDF page).

Target chunk ID: `eb0a11e3e37fb68aec2a5e03fca5bfa1f644327448a6b7b97d16b4ae2d1c8820`.

This identity is present in the clean-index manifest. An answer-bearing chunk therefore exists, but its membership in the clean index's RRF top five is unverified because saved top-five result IDs were unavailable. This is a report of historical saved text, not current electrical-safety guidance.

The app continues to use the original index. No answer-quality improvement or production fix is claimed.
