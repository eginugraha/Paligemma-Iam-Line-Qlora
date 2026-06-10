# SP-4 Svelte Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A SvelteKit web UI that uploads one handwriting-line image and streams the four HTR scenarios (M1–M4) into a side-by-side comparison, consuming the SP-2 `/v1/detect` NDJSON stream.

**Architecture:** A new `frontend/` SvelteKit + TypeScript app. The stream is consumed with `fetch()` + `ReadableStream` and an incremental NDJSON line parser; pure-TS units (`ndjson.ts`, `api.ts`, `types.ts`) hold all the logic and are unit-tested without a browser, while small scoped-CSS Svelte components render the upload area and the per-scenario columns (Layout A: upload on top, 4 columns below). RAG (M3/M4) is emitted by the backend only when `HTR_ENABLE_RAG=1`.

**Tech Stack:** SvelteKit 2, Svelte 5, TypeScript, Vite, Vitest (+ jsdom, @testing-library/svelte). Scoped CSS, no UI library. Node 18+ (dev machine has v25).

**Reference spec:** `docs/superpowers/specs/2026-06-11-sp4-svelte-frontend-design.md`

**Conventions:** small single-responsibility files; pure logic separated from components so it tests without a DOM; the live SP-2 backend already exposes everything (no backend changes). All commands run from `frontend/` unless noted.

---

## File Structure

```
frontend/
  package.json, svelte.config.js, vite.config.ts, tsconfig.json
  vitest-setup.ts
  src/app.html, src/app.d.ts
  src/lib/types.ts                     # event union + ScenarioId
  src/lib/ndjson.ts                    # parseNdjson(reader) — incremental NDJSON
  src/lib/api.ts                       # detectStream(file, gt?, base?) — POST + parse
  src/lib/components/ScenarioColumn.svelte
  src/lib/components/UploadArea.svelte
  src/lib/components/ImagePreview.svelte
  src/routes/+page.svelte              # Layout A orchestration
  src/lib/ndjson.test.ts, src/lib/api.test.ts, src/lib/components/ScenarioColumn.test.ts
```

---

## Task 1: Scaffold the SvelteKit + Vitest project

**Files (all new, under `frontend/`):** `package.json`, `svelte.config.js`, `vite.config.ts`, `tsconfig.json`, `vitest-setup.ts`, `src/app.html`, `src/app.d.ts`, `src/routes/+page.svelte`, `src/lib/sanity.test.ts`, `.gitignore`.

- [ ] **Step 1: Create the project files**

`frontend/package.json`:
```json
{
  "name": "htr-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite dev",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "check": "svelte-check --tsconfig ./tsconfig.json"
  },
  "devDependencies": {
    "@sveltejs/adapter-auto": "^3.3.1",
    "@sveltejs/kit": "^2.15.0",
    "@sveltejs/vite-plugin-svelte": "^5.0.0",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/svelte": "^5.2.4",
    "jsdom": "^25.0.1",
    "svelte": "^5.15.0",
    "svelte-check": "^4.1.0",
    "typescript": "^5.7.0",
    "vite": "^6.0.0",
    "vitest": "^2.1.8"
  }
}
```

`frontend/svelte.config.js`:
```js
import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: { adapter: adapter() }
};
export default config;
```

`frontend/vite.config.ts`:
```ts
import { sveltekit } from '@sveltejs/kit/vite';
import { svelteTesting } from '@testing-library/svelte/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit(), svelteTesting()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest-setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts}']
  }
});
```

`frontend/tsconfig.json`:
```json
{
  "extends": "./.svelte-kit/tsconfig.json",
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "sourceMap": true,
    "strict": true,
    "moduleResolution": "bundler"
  }
}
```

`frontend/vitest-setup.ts`:
```ts
import '@testing-library/jest-dom/vitest';
```

`frontend/src/app.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>HTR — Handwriting Recognition Compare</title>
    %sveltekit.head%
  </head>
  <body>
    <div>%sveltekit.body%</div>
  </body>
</html>
```

`frontend/src/app.d.ts`:
```ts
// See https://svelte.dev/docs/kit/types#app
declare global {
  namespace App {}
}
export {};
```

`frontend/src/routes/+page.svelte`:
```svelte
<h1>HTR Compare</h1>
<p>Frontend scaffold OK.</p>
```

`frontend/src/lib/sanity.test.ts`:
```ts
import { describe, it, expect } from 'vitest';

describe('toolchain', () => {
  it('runs vitest', () => {
    expect(1 + 1).toBe(2);
  });
});
```

`frontend/.gitignore`:
```
node_modules/
/build
/.svelte-kit
/package-lock.json
.env
.env.*
!.env.example
```

- [ ] **Step 2: Install dependencies and sync SvelteKit**

Run (from `frontend/`):
```bash
npm install
npx svelte-kit sync
```
Expected: dependencies install; `.svelte-kit/` is generated (creates the tsconfig base). If `npm install` reports a peer-dependency conflict, let npm resolve it or bump the conflicting dev-dependency to its nearest compatible minor — the goal is a clean install and a green `npm test`. Report BLOCKED only if it cannot be made to install + test.

- [ ] **Step 3: Run the sanity test**

Run: `npm test`
Expected: PASS — `sanity.test.ts` runs green (1 test passed). This proves Vitest + jsdom + the Svelte testing plugin are wired before any real code is written.

- [ ] **Step 4: Commit**

```bash
cd .. && git add frontend && git commit -m "feat(sp4): scaffold SvelteKit + Vitest frontend"
```

---

## Task 2: Event types + NDJSON parser

**Files:**
- Create: `frontend/src/lib/types.ts`
- Create: `frontend/src/lib/ndjson.ts`
- Test: `frontend/src/lib/ndjson.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/lib/ndjson.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { parseNdjson } from './ndjson';

// Build a ReadableStreamDefaultReader-like object from string chunks.
function readerFrom(chunks: string[]): ReadableStreamDefaultReader<Uint8Array> {
  const enc = new TextEncoder();
  let i = 0;
  return {
    read: async () =>
      i < chunks.length
        ? { done: false, value: enc.encode(chunks[i++]) }
        : { done: true, value: undefined }
  } as unknown as ReadableStreamDefaultReader<Uint8Array>;
}

async function collect(chunks: string[]): Promise<unknown[]> {
  const out: unknown[] = [];
  for await (const obj of parseNdjson(readerFrom(chunks))) out.push(obj);
  return out;
}

describe('parseNdjson', () => {
  it('parses one object per line', async () => {
    const out = await collect(['{"a":1}\n{"a":2}\n']);
    expect(out).toEqual([{ a: 1 }, { a: 2 }]);
  });

  it('joins a line split across chunks', async () => {
    const out = await collect(['{"ev":"me', 'ta"}\n']);
    expect(out).toEqual([{ ev: 'meta' }]);
  });

  it('handles multiple lines in one chunk and a missing trailing newline', async () => {
    const out = await collect(['{"a":1}\n{"a":2}', '\n{"a":3}']);
    expect(out).toEqual([{ a: 1 }, { a: 2 }, { a: 3 }]);
  });

  it('ignores blank lines', async () => {
    const out = await collect(['\n{"a":1}\n\n']);
    expect(out).toEqual([{ a: 1 }]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- ndjson`
Expected: FAIL — cannot resolve `./ndjson` (module not found).

- [ ] **Step 3: Write minimal implementation**

`frontend/src/lib/types.ts`:
```ts
/** Scenario identifiers in the NDJSON stream (matches the SP-2 backend). */
export type ScenarioId = 'm1' | 'm2' | 'm3' | 'm4';

export interface MetaEvent {
  event: 'meta';
  filename: string;
  has_ground_truth: boolean;
}

export interface ResultEvent {
  event: 'result';
  model: ScenarioId;
  text: string;
  cer: number | null; // null when no ground truth was supplied
  wer: number | null;
  latency_seconds: number;
  log: string;
  status_tag: string; // "Raw Output" | "Reasoned" | "Corrected" | "Optimal"
}

export interface ErrorEvent {
  event: 'error';
  model: ScenarioId;
  message: string;
}

export interface DoneEvent {
  event: 'done';
}

export type DetectEvent = MetaEvent | ResultEvent | ErrorEvent | DoneEvent;
```

`frontend/src/lib/ndjson.ts`:
```ts
/**
 * Incrementally parse a byte stream of newline-delimited JSON (NDJSON).
 *
 * The SP-2 /v1/detect endpoint streams one JSON object per line. We decode chunks as they
 * arrive, buffer any partial trailing line across chunk boundaries, and yield each complete
 * line as a parsed object — so the UI can render scenario columns the moment their event lands
 * rather than waiting for the whole response.
 */
export async function* parseNdjson(
  reader: ReadableStreamDefaultReader<Uint8Array>
): AsyncGenerator<unknown> {
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let nl: number;
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) yield JSON.parse(line);
    }
  }

  // Flush any final line that had no trailing newline.
  buffer += decoder.decode();
  const last = buffer.trim();
  if (last) yield JSON.parse(last);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- ndjson`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd .. && git add frontend/src/lib/types.ts frontend/src/lib/ndjson.ts frontend/src/lib/ndjson.test.ts && git commit -m "feat(sp4): event types + incremental NDJSON parser"
```

---

## Task 3: `detectStream` API client

**Files:**
- Create: `frontend/src/lib/api.ts`
- Test: `frontend/src/lib/api.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/lib/api.test.ts`:
```ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { detectStream } from './api';
import type { DetectEvent } from './types';

function bodyFrom(lines: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const l of lines) controller.enqueue(enc.encode(l));
      controller.close();
    }
  });
}

async function collect(file: File, gt?: string): Promise<DetectEvent[]> {
  const out: DetectEvent[] = [];
  for await (const e of detectStream(file, gt, 'http://test')) out.push(e);
  return out;
}

afterEach(() => vi.restoreAllMocks());

const PNG = new File([new Uint8Array([1, 2, 3])], 'line.png', { type: 'image/png' });

describe('detectStream', () => {
  it('POSTs multipart and yields parsed events in order', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      body: bodyFrom([
        '{"event":"meta","filename":"line.png","has_ground_truth":true}\n',
        '{"event":"result","model":"m1","text":"hi","cer":0,"wer":0,"latency_seconds":0.5,"log":"x","status_tag":"Raw Output"}\n',
        '{"event":"done"}\n'
      ])
    }));
    vi.stubGlobal('fetch', fetchMock);

    const events = await collect(PNG, 'hi');

    expect(events.map((e) => e.event)).toEqual(['meta', 'result', 'done']);
    // POSTed to the right URL with multipart FormData carrying the file + ground_truth.
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('http://test/v1/detect');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('file')).toBeInstanceOf(File);
    expect((init.body as FormData).get('ground_truth')).toBe('hi');
  });

  it('omits ground_truth when not provided', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, body: bodyFrom(['{"event":"done"}\n']) }));
    vi.stubGlobal('fetch', fetchMock);
    await collect(PNG);
    expect((fetchMock.mock.calls[0][1].body as FormData).get('ground_truth')).toBeNull();
  });

  it('throws on a non-OK response (e.g. 422)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 422, body: null })));
    await expect(collect(PNG)).rejects.toThrow(/422/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- api`
Expected: FAIL — cannot resolve `./api`.

- [ ] **Step 3: Write minimal implementation**

`frontend/src/lib/api.ts`:
```ts
import { parseNdjson } from './ndjson';
import type { DetectEvent } from './types';

/** Backend base URL. Override at build/dev time with VITE_API_BASE. */
const DEFAULT_BASE: string =
  (import.meta.env?.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000';

/**
 * POST an image (and optional ground truth) to /v1/detect and yield each NDJSON event as it
 * streams back. Throws on a non-OK HTTP status (e.g. 422 for an undecodable image) before any
 * streaming begins, so callers can show a top-level error.
 */
export async function* detectStream(
  file: File,
  groundTruth?: string,
  baseUrl: string = DEFAULT_BASE
): AsyncGenerator<DetectEvent> {
  const form = new FormData();
  form.append('file', file);
  if (groundTruth) form.append('ground_truth', groundTruth);

  const res = await fetch(`${baseUrl}/v1/detect`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`detect failed: HTTP ${res.status}`);
  if (!res.body) throw new Error('detect failed: empty response body');

  yield* parseNdjson(res.body.getReader()) as AsyncGenerator<DetectEvent>;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- api`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd .. && git add frontend/src/lib/api.ts frontend/src/lib/api.test.ts && git commit -m "feat(sp4): detectStream API client (multipart POST + NDJSON)"
```

---

## Task 4: `ScenarioColumn` component

**Files:**
- Create: `frontend/src/lib/components/ScenarioColumn.svelte`
- Test: `frontend/src/lib/components/ScenarioColumn.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/lib/components/ScenarioColumn.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import ScenarioColumn from './ScenarioColumn.svelte';
import type { ResultEvent } from '$lib/types';

const RESULT: ResultEvent = {
  event: 'result',
  model: 'm3',
  text: 'the quick brown fox',
  cer: 0,
  wer: 0,
  latency_seconds: 1.1,
  log: 'pgvector: fux -> fox',
  status_tag: 'Corrected'
};

describe('ScenarioColumn', () => {
  it('renders the filled state with text, badge, latency, log and metrics', () => {
    render(ScenarioColumn, {
      props: { id: 'm3', title: 'QLoRA + RAG', state: 'filled', result: RESULT, hasGroundTruth: true }
    });
    expect(screen.getByText('the quick brown fox')).toBeInTheDocument();
    expect(screen.getByText('Corrected')).toBeInTheDocument();
    expect(screen.getByText(/1\.1/)).toBeInTheDocument();
    expect(screen.getByText(/pgvector: fux -> fox/)).toBeInTheDocument();
  });

  it('shows an em dash for CER/WER when there is no ground truth', () => {
    render(ScenarioColumn, {
      props: { id: 'm3', title: 'QLoRA + RAG', state: 'filled', result: { ...RESULT, cer: null, wer: null }, hasGroundTruth: false }
    });
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);
  });

  it('shows a spinner while pending', () => {
    render(ScenarioColumn, { props: { id: 'm1', title: 'Baseline', state: 'pending' } });
    expect(screen.getByTestId('spinner')).toBeInTheDocument();
  });

  it('shows the message when in the error state', () => {
    render(ScenarioColumn, { props: { id: 'm2', title: 'CoT', state: 'error', errorMessage: 'engine timeout' } });
    expect(screen.getByText(/engine timeout/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- ScenarioColumn`
Expected: FAIL — cannot resolve `./ScenarioColumn.svelte`.

- [ ] **Step 3: Write minimal implementation**

`frontend/src/lib/components/ScenarioColumn.svelte`:
```svelte
<script lang="ts">
  import type { ResultEvent } from '$lib/types';

  let {
    id,
    title,
    state,
    result,
    errorMessage = '',
    hasGroundTruth = false
  }: {
    id: string;
    title: string;
    state: 'pending' | 'filled' | 'error' | 'disabled';
    result?: ResultEvent;
    errorMessage?: string;
    hasGroundTruth?: boolean;
  } = $props();

  // status_tag -> badge colour class (one fixed mapping for the four scenarios).
  const BADGE: Record<string, string> = {
    'Raw Output': 'b1',
    Reasoned: 'b2',
    Corrected: 'b3',
    Optimal: 'b4'
  };
  const badgeClass = $derived(result ? (BADGE[result.status_tag] ?? 'b1') : 'b1');
  const metric = (v: number | null) => (v === null || !hasGroundTruth ? '—' : v.toFixed(2));
</script>

<div class="col" class:err={state === 'error'} class:dis={state === 'disabled'}>
  <div class="head">
    <span class="id">{id.toUpperCase()}</span>
    <span class="title">{title}</span>
  </div>

  {#if state === 'pending'}
    <div class="spinner" data-testid="spinner" aria-label="loading"></div>
    <p class="muted center">running…</p>
  {:else if state === 'filled' && result}
    <span class="badge {badgeClass}">{result.status_tag}</span>
    <p class="text">{result.text}</p>
    <p class="row"><b>CER</b> {metric(result.cer)} &nbsp; <b>WER</b> {metric(result.wer)}</p>
    <p class="row">⏱ {result.latency_seconds.toFixed(2)}s</p>
    <pre class="log">{result.log}</pre>
  {:else if state === 'error'}
    <span class="badge berr">Failed</span>
    <p class="text muted">⚠ {errorMessage}</p>
  {:else}
    <p class="muted center">not run — enable RAG (HTR_ENABLE_RAG=1)</p>
  {/if}
</div>

<style>
  .col { flex: 1; min-width: 0; background: #fff; border: 1px solid #e1e6ef; border-radius: 9px; padding: 12px; }
  .col.err { border-color: #f0c9c9; background: #fdf6f6; }
  .col.dis { background: #f7f9fc; }
  .head { display: flex; align-items: baseline; gap: 6px; margin-bottom: 8px; }
  .id { font-weight: 800; color: #111827; }
  .title { font-size: 12px; color: #94a3b8; }
  .badge { display: inline-block; font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 999px; color: #fff; }
  .b1 { background: #6b7280; } .b2 { background: #7c3aed; } .b3 { background: #2563eb; } .b4 { background: #16a34a; } .berr { background: #dc2626; }
  .text { font-weight: 700; font-size: 15px; color: #1f2937; margin: 8px 0; overflow-wrap: anywhere; }
  .row { font-size: 12px; color: #475569; margin: 3px 0; }
  .row b { color: #111827; }
  .log { font-size: 11px; color: #475569; background: #eef2f8; border-radius: 6px; padding: 8px; margin-top: 8px; white-space: pre-wrap; overflow-wrap: anywhere; font-family: ui-monospace, monospace; }
  .muted { color: #94a3b8; }
  .center { text-align: center; }
  .spinner { width: 22px; height: 22px; border: 3px solid #d7deea; border-top-color: #2563eb; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 14px auto; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- ScenarioColumn`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd .. && git add frontend/src/lib/components/ScenarioColumn.svelte frontend/src/lib/components/ScenarioColumn.test.ts && git commit -m "feat(sp4): ScenarioColumn component (pending/filled/error/disabled)"
```

---

## Task 5: `UploadArea` + `ImagePreview` components

**Files:**
- Create: `frontend/src/lib/components/UploadArea.svelte`
- Create: `frontend/src/lib/components/ImagePreview.svelte`
- Test: `frontend/src/lib/components/UploadArea.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/lib/components/UploadArea.test.ts`:
```ts
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import UploadArea from './UploadArea.svelte';

describe('UploadArea', () => {
  it('emits the selected image file via onfile', async () => {
    const onfile = vi.fn();
    render(UploadArea, { props: { onfile, disabled: false } });

    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const file = new File([new Uint8Array([1])], 'line.png', { type: 'image/png' });
    await fireEvent.change(input, { target: { files: [file] } });

    expect(onfile).toHaveBeenCalledOnce();
    expect(onfile.mock.calls[0][0]).toBe(file);
  });

  it('rejects a non-image file with an inline message and does not emit', async () => {
    const onfile = vi.fn();
    render(UploadArea, { props: { onfile, disabled: false } });

    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const bad = new File(['x'], 'notes.txt', { type: 'text/plain' });
    await fireEvent.change(input, { target: { files: [bad] } });

    expect(onfile).not.toHaveBeenCalled();
    expect(screen.getByText(/png, jpg, or jpeg/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- UploadArea`
Expected: FAIL — cannot resolve `./UploadArea.svelte`.

- [ ] **Step 3: Write minimal implementation**

`frontend/src/lib/components/UploadArea.svelte`:
```svelte
<script lang="ts">
  let {
    onfile,
    disabled = false
  }: { onfile: (file: File) => void; disabled?: boolean } = $props();

  let error = $state('');
  const OK = ['image/png', 'image/jpeg'];

  function accept(file: File | undefined) {
    if (!file) return;
    if (!OK.includes(file.type)) {
      error = 'Please choose a png, jpg, or jpeg image.';
      return;
    }
    error = '';
    onfile(file);
  }

  function onChange(e: Event) {
    accept((e.currentTarget as HTMLInputElement).files?.[0]);
  }
  function onDrop(e: DragEvent) {
    e.preventDefault();
    if (disabled) return;
    accept(e.dataTransfer?.files?.[0]);
  }
</script>

<div
  class="drop"
  class:disabled
  role="button"
  tabindex="0"
  ondragover={(e) => e.preventDefault()}
  ondrop={onDrop}
>
  <p>⬆ Drag &amp; drop a handwriting-line image here, or</p>
  <label class="btn">
    Choose file
    <input
      data-testid="file-input"
      type="file"
      accept="image/png,image/jpeg"
      {disabled}
      onchange={onChange}
      hidden
    />
  </label>
  {#if error}<p class="error">{error}</p>{/if}
</div>

<style>
  .drop { border: 2px dashed #9aa7bd; border-radius: 10px; padding: 22px; text-align: center; background: #f6f8fc; color: #5b6b86; }
  .drop.disabled { opacity: 0.6; }
  .btn { display: inline-block; margin-top: 8px; background: #2563eb; color: #fff; padding: 7px 14px; border-radius: 7px; font-weight: 600; cursor: pointer; }
  .error { color: #dc2626; margin-top: 8px; font-size: 13px; }
</style>
```

`frontend/src/lib/components/ImagePreview.svelte`:
```svelte
<script lang="ts">
  let { file }: { file: File | null } = $props();
  // Build/revoke an object URL whenever the file changes (no leak).
  let url = $state('');
  $effect(() => {
    if (!file) {
      url = '';
      return;
    }
    const u = URL.createObjectURL(file);
    url = u;
    return () => URL.revokeObjectURL(u);
  });
</script>

{#if url}
  <figure class="preview">
    <img src={url} alt="uploaded handwriting line" />
    <figcaption>{file?.name}</figcaption>
  </figure>
{/if}

<style>
  .preview { margin: 0; }
  .preview img { max-height: 90px; border: 1px solid #e1e6ef; border-radius: 6px; background: #fff; }
  figcaption { font-size: 12px; color: #94a3b8; margin-top: 4px; }
</style>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- UploadArea`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd .. && git add frontend/src/lib/components/UploadArea.svelte frontend/src/lib/components/ImagePreview.svelte frontend/src/lib/components/UploadArea.test.ts && git commit -m "feat(sp4): UploadArea (validated) + ImagePreview components"
```

---

## Task 6: `+page.svelte` — Layout A orchestration

**Files:**
- Modify: `frontend/src/routes/+page.svelte` (replace the scaffold placeholder)

This wires the components and the stream together. It is verified by the full unit suite staying
green plus a manual run against the live backend (the streaming glue is integration code; its
units are already tested in Tasks 2–5).

- [ ] **Step 1: Replace the page**

`frontend/src/routes/+page.svelte`:
```svelte
<script lang="ts">
  import UploadArea from '$lib/components/UploadArea.svelte';
  import ImagePreview from '$lib/components/ImagePreview.svelte';
  import ScenarioColumn from '$lib/components/ScenarioColumn.svelte';
  import { detectStream } from '$lib/api';
  import type { DetectEvent, ResultEvent, ScenarioId } from '$lib/types';

  const SCENARIOS: { id: ScenarioId; title: string }[] = [
    { id: 'm1', title: 'Baseline QLoRA' },
    { id: 'm2', title: 'QLoRA + CoT' },
    { id: 'm3', title: 'QLoRA + RAG' },
    { id: 'm4', title: 'Hybrid CoT + RAG' }
  ];

  type ColState = 'idle' | 'pending' | 'filled' | 'error' | 'disabled';
  interface Col { state: ColState; result?: ResultEvent; errorMessage?: string }

  let file = $state<File | null>(null);
  let groundTruth = $state('');
  let running = $state(false);
  let topError = $state('');
  let hasGroundTruth = $state(false);
  let cols = $state<Record<ScenarioId, Col>>({
    m1: { state: 'idle' }, m2: { state: 'idle' }, m3: { state: 'idle' }, m4: { state: 'idle' }
  });

  function onfile(f: File) {
    file = f;
  }

  async function run() {
    if (!file || running) return;
    running = true;
    topError = '';
    hasGroundTruth = groundTruth.trim().length > 0;
    // All four columns start pending; any that never arrive become "disabled" at done.
    cols = { m1: { state: 'pending' }, m2: { state: 'pending' }, m3: { state: 'pending' }, m4: { state: 'pending' } };

    try {
      for await (const ev of detectStream(file, groundTruth.trim() || undefined)) {
        apply(ev);
      }
    } catch (e) {
      topError = e instanceof Error ? e.message : String(e);
      cols = { m1: { state: 'idle' }, m2: { state: 'idle' }, m3: { state: 'idle' }, m4: { state: 'idle' } };
    } finally {
      running = false;
    }
  }

  function apply(ev: DetectEvent) {
    if (ev.event === 'meta') {
      hasGroundTruth = ev.has_ground_truth;
    } else if (ev.event === 'result') {
      cols[ev.model] = { state: 'filled', result: ev };
    } else if (ev.event === 'error') {
      cols[ev.model] = { state: 'error', errorMessage: ev.message };
    } else if (ev.event === 'done') {
      // Any column still pending never streamed (RAG off / skipped) -> mark disabled.
      for (const { id } of SCENARIOS) {
        if (cols[id].state === 'pending') cols[id] = { state: 'disabled' };
      }
    }
  }
</script>

<main>
  <h1>HTR — Handwriting Recognition Compare</h1>
  <p class="sub">Upload one handwriting line; compare M1–M4 side by side.</p>

  <section class="controls">
    <UploadArea {onfile} disabled={running} />
    <div class="meta">
      <ImagePreview {file} />
      <label class="gt">
        Ground truth (optional)
        <input type="text" bind:value={groundTruth} placeholder="the quick brown fox" disabled={running} />
      </label>
      <button class="run" onclick={run} disabled={!file || running}>
        {running ? 'Running…' : '▶ Run M1–M4'}
      </button>
    </div>
  </section>

  {#if topError}<p class="top-error">⚠ {topError}</p>{/if}

  <section class="cols">
    {#each SCENARIOS as s (s.id)}
      {#if cols[s.id].state !== 'idle'}
        <ScenarioColumn
          id={s.id}
          title={s.title}
          state={cols[s.id].state as 'pending' | 'filled' | 'error' | 'disabled'}
          result={cols[s.id].result}
          errorMessage={cols[s.id].errorMessage}
          {hasGroundTruth}
        />
      {/if}
    {/each}
  </section>
</main>

<style>
  :global(body) { margin: 0; font-family: system-ui, sans-serif; background: #eef2f8; color: #1f2937; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 4px; font-size: 22px; }
  .sub { margin: 0 0 18px; color: #64748b; }
  .controls { display: flex; flex-direction: column; gap: 12px; background: #fff; border: 1px solid #e1e6ef; border-radius: 12px; padding: 16px; }
  .meta { display: flex; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
  .gt { display: flex; flex-direction: column; font-size: 12px; color: #64748b; gap: 4px; flex: 1; min-width: 200px; }
  .gt input { padding: 8px; border: 1px solid #cbd5e1; border-radius: 7px; font-size: 14px; }
  .run { background: #2563eb; color: #fff; border: 0; padding: 10px 18px; border-radius: 8px; font-weight: 700; cursor: pointer; }
  .run:disabled { opacity: 0.5; cursor: default; }
  .top-error { color: #dc2626; font-weight: 600; }
  .cols { display: flex; gap: 12px; margin-top: 18px; align-items: stretch; }
</style>
```

- [ ] **Step 2: Run the full unit suite (no regressions)**

Run (from `frontend/`): `npm test`
Expected: PASS — all unit tests from Tasks 2–5 stay green.

- [ ] **Step 3: Type-check**

Run: `npm run check`
Expected: 0 errors (warnings acceptable).

- [ ] **Step 4: Manual end-to-end smoke (against the live backend)**

In one terminal (repo root): `HTR_ENABLE_RAG=1 uvicorn htr_sp2.api:app --app-dir src`
In another (`frontend/`): `npm run dev`, open the printed URL, upload a handwriting PNG, set a ground truth, click Run.
Expected: four columns stream in (m1, m2, m3, m4 when RAG is on); CER/WER show when ground truth is set; the RAG log appears in m3/m4. (Backend on the default `http://localhost:8000`; if it differs, set `VITE_API_BASE` in `frontend/.env`.)

- [ ] **Step 5: Commit**

```bash
cd .. && git add frontend/src/routes/+page.svelte && git commit -m "feat(sp4): detect/compare page (Layout A) wiring upload + streaming columns"
```

---

## Post-implementation (manual)

- Add `frontend/.env.example` with `VITE_API_BASE=http://localhost:8000` if deploying elsewhere.
- The global statistics dashboard (FR-FE-05) is SP-5.

---

## Self-Review

- **Spec coverage:** §2 SvelteKit+TS+scoped CSS+Vitest (Task 1); FR-FE-01 upload drag-drop + validation (Task 5 UploadArea); image preview (Task 5 ImagePreview); §3 NDJSON contract → types + parser (Task 2); stream consumption via fetch+getReader (Task 3 api); §4/§6 ScenarioColumn 5 params + pending/filled/error/disabled + always-visible log (Task 4); Layout A + progressive streaming + per-scenario error isolation + no-GT "—" + RAG-off disabled (Task 6 +page); §7 Vitest unit tests for ndjson/api/ScenarioColumn/UploadArea (Tasks 2–5). FR-FE-05 dashboard explicitly deferred (§8). All spec sections map to a task.
- **Placeholders:** none — every step ships runnable code/tests and exact commands. The one non-deterministic point (npm peer-dep resolution) is called out with explicit guidance in Task 1.
- **Type consistency:** `DetectEvent`/`ResultEvent`/`ScenarioId` (Task 2) are imported unchanged by `api.ts` (Task 3), `ScenarioColumn` (Task 4), and `+page.svelte` (Task 6); `detectStream(file, groundTruth?, baseUrl?)` (Task 3) is called with the same shape in Task 6; `ScenarioColumn` props (`id,title,state,result?,errorMessage?,hasGroundTruth`) match between its definition (Task 4) and use (Task 6); `UploadArea` `onfile(file)` / `disabled` props match between Task 5 and Task 6.
```
