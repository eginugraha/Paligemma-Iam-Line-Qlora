# `data/` — vendored datasets

## `english_words.txt`

A general English wordlist (one lowercased word per line, ~370k words **including inflected
forms** like `sings`, `stars`, `groups`). Used by `src/htr_sp3/english_words.py` as the SP-3
corrector's **validity gate** (Option B — see `docs/sp3-rag-correction-fix-2026-06-15.md`).

**Why vendored (not downloaded at runtime / read from `/usr/share/dict/words`):**
reproducibility. The exact gate is captured in version control and is byte-identical on every
machine (laptop, server, RunPod), the same philosophy as the pinned `requirements.txt`. The
macOS/Unix `web2` list was rejected because it is lemma-based (has `sing` but not `sings`), which
defeats the gate's purpose.

**Source & provenance:** derived from the public-domain
[dwyl/english-words](https://github.com/dwyl/english-words) `words_alpha.txt` (370,105 entries),
then normalized: lowercased, stripped, kept only single alphabetic words (regex
`[a-z]+(?:'[a-z]+)*`), deduped, and sorted. Regenerate with the snippet in that report.

**Anti-leakage note:** this wordlist widens only the *gate* (which words are left untouched). The
correction *candidate store* (pgvector) stays IAM-**train**-only. A general dictionary is external
knowledge, not test/validation labels, so using it introduces no leakage into the reported numbers.
