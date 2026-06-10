# SP-4 — Svelte Frontend (detect / compare page) — Design

**Date:** 2026-06-11
**Status:** Approved (design)
**Depends on:** SP-2 (`POST /v1/detect` NDJSON stream, CORS enabled), SP-3 (M3/M4 emitted when RAG is on).
**Blocks:** SP-5 (global statistics dashboard — separate sub-project).

---

## 1. Purpose

A Svelte web UI that lets a user upload one handwriting-line image and compare the four HTR
scenarios (M1–M4) side by side, with results streaming in column-by-column. Implements the
PRD's front-end requirements FR-FE-01..04. The global batch-evaluation dashboard (FR-FE-05) is
deferred to SP-5.

## 2. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Framework | **SvelteKit** + TypeScript |
| Styling | **Svelte scoped CSS** (no UI library / no Tailwind) |
| Page layout | **Layout A** — full-width upload + image + ground-truth on top; the 4 scenario columns side-by-side below |
| Stream consumption | `fetch()` + `response.body.getReader()`, parsed as NDJSON incrementally (POST multipart rules out EventSource) |
| Per-column log | **Always visible** (full CoT reasoning / pgvector match log shown in the column) |
| Column states | **pending (spinner) → filled / error** |
| Testing | **Vitest** unit tests (no Playwright) |
| Dashboard (FR-FE-05) | **Deferred to SP-5** (may add a placeholder nav link) |
| Backend base URL | env `VITE_API_BASE` (default `http://localhost:8000`) |
| Location | new `frontend/` directory at the repo root |

## 3. The API contract this consumes

`POST /v1/detect` (multipart: `file`, optional `ground_truth`) → `application/x-ndjson` stream,
one JSON object per line:

- `{"event":"meta","filename":...,"has_ground_truth":bool}` — first line.
- `{"event":"result","model":"m1|m2|m3|m4","text":...,"cer":num|null,"wer":num|null,"latency_seconds":num,"log":str,"status_tag":str}` — one per successful scenario.
- `{"event":"error","model":...,"message":str}` — one per failed scenario; stream continues.
- `{"event":"done"}` — final line.

Scenarios stream in order m1, m2, then (when RAG enabled) m3, m4. CER/WER are `null` when no
ground truth was supplied. (The PRD's single-JSON `results{}` shape is superseded by this
streaming NDJSON contract, which the live SP-2 backend actually emits.)

## 4. Architecture & components

Small, single-purpose units with clear interfaces:

```
frontend/
  src/lib/types.ts            # discriminated-union event types + ScenarioId = "m1".."m4"
  src/lib/ndjson.ts           # parseNdjson(reader) -> AsyncGenerator<object> (PURE wrt fetch)
  src/lib/api.ts              # detectStream(file, groundTruth?, baseUrl?) -> AsyncGenerator<Event>
  src/lib/components/
    UploadArea.svelte         # drag-drop + file button + ground-truth textarea (FR-FE-01)
    ImagePreview.svelte       # shows the chosen image (object URL)
    ScenarioColumn.svelte     # one column: badge + 5 params; renders pending/filled/error
  src/routes/+page.svelte     # Layout A; owns state, drives detectStream, routes events to columns
  tests/                      # Vitest unit tests (ndjson, api, ScenarioColumn)
```

- **`ndjson.ts`** — `parseNdjson(reader)`: reads a `ReadableStreamDefaultReader<Uint8Array>`,
  decodes UTF-8, buffers partial lines across chunks, and yields each parsed JSON object as a
  full line completes. The trickiest unit; fully testable from a fake reader.
- **`api.ts`** — `detectStream(file, groundTruth?, baseUrl?)`: builds the multipart `FormData`,
  POSTs to `${baseUrl}/v1/detect`, and delegates to `parseNdjson(res.body.getReader())`, yielding
  typed events. Throws a typed error on non-OK status (e.g. 422 invalid image). Testable with a
  mocked `fetch`.
- **`ScenarioColumn.svelte`** — props: `id` ("m1".."m4"), `title`, `state` ("pending"|"filled"|
  "error"|"disabled"), `result?`, `errorMessage?`, `hasGroundTruth`. Pure presentation: badge
  from `status_tag`, the 5 params (text; CER/WER or "—" when no GT; latency; always-visible log;
  status), spinner when pending, error styling when error, a muted "not run (enable RAG)" note
  when disabled.
- **`+page.svelte`** — holds `columns` keyed by m1..m4 (all start `pending` once a run begins),
  `meta`, and `running`. On submit: show preview, iterate `detectStream`, and on each event set
  `meta` / fill the matching column / mark it error / clear `running` on `done`.

## 5. Data flow

1. User drops/selects an image (validated client-side: png/jpg/jpeg) and optionally types a
   ground truth.
2. On "Run": all four columns enter **pending** (spinner); preview shows.
3. `detectStream` POSTs and yields events: `meta` populates the header (filename, GT flag);
   each `result` fills its column; each `error` marks its column; `done` ends the run.
4. Columns therefore appear progressively (m1, then m2, then m3, m4) — matching FR-FE-02.

## 6. States & error handling

- **Pre-flight:** reject non-image / wrong extension before sending; show an inline message.
- **HTTP 422** (undecodable image from backend): caught from `detectStream`, shown as a top-level
  error; columns reset.
- **Per-scenario `error` event:** that column shows the **error** state with the message; the
  others keep working and the stream still ends with `done` (SP-2 isolation preserved).
- **No ground truth:** CER/WER render as "—" (the events carry `null`).
- **RAG off (backend default):** only m1/m2 arrive; m3/m4 columns stay pending then are marked
  "not run" when `done` arrives without them (shown as a muted "disabled — enable RAG" note).

## 7. Testing (Vitest, no browser/network)

- `ndjson.test.ts` — feeds a fake reader chunked at arbitrary byte boundaries (incl. a line split
  across two chunks, multiple lines in one chunk, trailing newline) → asserts the exact sequence
  of parsed objects.
- `api.test.ts` — mocks `fetch` to return a `ReadableStream` of canned NDJSON; asserts
  `detectStream` yields meta→result×N→done in order, builds the right `FormData`, and throws on a
  non-OK status.
- `ScenarioColumn.test.ts` (@testing-library/svelte) — renders each state: pending shows a
  spinner; filled shows text + the right badge + latency + log + CER/WER (and "—" when
  `hasGroundTruth=false`); error shows the message.

## 8. Out of scope (deferred)

- **FR-FE-05 global statistics dashboard** → SP-5 (needs batch-eval data). A disabled nav link
  may hint at it.
- GGUF/local engine, batch evaluation, authentication, pixel-perfect visual polish.
- Backend changes (SP-2/SP-3 already expose everything needed).
