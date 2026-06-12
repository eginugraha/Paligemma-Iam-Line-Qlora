<script lang="ts">
  /**
   * /history — FR-FE-06: End-User Upload History
   *
   * Shows the user a paginated list of all images they have uploaded via the
   * /v1/detect endpoint, along with per-scenario HTR results for each one.
   *
   * Data flow:
   *   1. On mount, `fetchUploads()` loads the upload list (newest-first).
   *   2. Each row shows a thumbnail, filename, and formatted timestamp.
   *   3. Clicking a row toggles an expandable detail panel that reveals
   *      per-scenario metrics (CER, WER, latency, log, status_tag).
   *   4. Clicking the same row again collapses the panel (toggle behaviour).
   *
   * Svelte 5 runes used:
   *   $state   — mutable reactive variables (uploads, expanded, error).
   *   $effect  — async-IIFE initial data load on mount (no re-run after deps change,
   *              because all reactive reads occur after the first await boundary).
   *
   * The detail panel uses `{#if expanded === u.id}` so the log content is NOT
   * inserted into the DOM until the user actually clicks the row (a11y + perf).
   */

  import { fetchUploads, uploadImageUrl } from '$lib/api';
  import type { UploadRecord } from '$lib/types';

  // ─── Scenario label map ───────────────────────────────────────────────────────

  /**
   * Maps internal scenario ids (m1–m4) to human-readable thesis labels.
   * Used in the expanded detail panel to annotate each scenario's result block.
   */
  const SCENARIO_LABEL: Record<string, string> = {
    m1: 'M1 QLoRA',
    m2: 'M2 +CoT',
    m3: 'M3 +RAG',
    m4: 'M4 Hybrid'
  };

  // ─── State ────────────────────────────────────────────────────────────────────

  /** Full list of upload records returned by the backend (newest-first). */
  let uploads = $state<UploadRecord[]>([]);

  /**
   * The id of the currently expanded upload row, or `null` when all rows are
   * collapsed.  Clicking a row with this id collapses it; clicking any other row
   * expands it and collapses the previously expanded one.
   */
  let expanded = $state<number | null>(null);

  /** Non-empty string when the initial fetch fails; displayed as an error banner. */
  let error = $state('');

  // ─── Data loading ─────────────────────────────────────────────────────────────

  /**
   * Fetch the upload list once on mount.
   *
   * The async IIFE pattern ensures $effect runs exactly once: Svelte 5 tracks
   * reactive reads that happen synchronously before the first `await`.  Because we
   * hand off immediately to the IIFE, all reads happen asynchronously — Svelte
   * never registers a dependency and the effect never re-runs.
   */
  $effect(() => {
    (async () => {
      try {
        uploads = await fetchUploads();
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
      }
    })();
  });

  // ─── Helpers ─────────────────────────────────────────────────────────────────

  /**
   * Toggle the expanded detail panel for the given upload id.
   * Clicking the already-expanded row sets `expanded` back to `null` (collapse).
   * Clicking any other row replaces the current expanded id with the new one.
   *
   * @param id - The upload's primary key.
   */
  function toggle(id: number): void {
    expanded = expanded === id ? null : id;
  }

  /**
   * Format a nullable percentage metric for display.
   * Returns an em-dash when the value is null or undefined (no ground truth).
   *
   * @param v - The percentage value (0–100), or null/undefined.
   */
  function pct(v: number | null | undefined): string {
    return v == null ? '—' : `${v}%`;
  }
</script>

<!--
  History page — one section:
    1. Error banner (when fetch fails).
    2. Empty state message (when list is empty and no error).
    3. Upload list — each row is a <button> with thumbnail, filename, timestamp.
       Clicking a row toggles an expandable per-scenario detail panel.
-->

<div class="history">
  <h2>Upload History</h2>

  <!-- ── Error banner ─────────────────────────────────────────────────────── -->
  {#if error}
    <p class="error" role="alert">⚠ {error}</p>
  {/if}

  <!-- ── Empty-state message ──────────────────────────────────────────────── -->
  {#if uploads.length === 0 && !error}
    <p class="empty">No uploads yet.</p>
  {/if}

  <!-- ── Upload list ──────────────────────────────────────────────────────── -->
  <!--
    Each upload is rendered as a <button> for keyboard accessibility (focusable,
    activatable with Enter/Space) and correct role semantics.  The detail panel
    sits inside the same list item, below the summary row, and is guarded by an
    {#if} so its DOM nodes only exist when the row is expanded.
  -->
  <ul class="upload-list">
    {#each uploads as u (u.id)}
      <li class="upload-item">
        <!-- ── Row button — clicking toggles the detail panel ────────────── -->
        <!--
          The entire summary row is wrapped in a <button> so the user can click
          anywhere on it (thumbnail, filename, or timestamp) to expand/collapse.
          The aria-expanded attribute communicates the current toggle state to
          assistive technologies.
        -->
        <button
          class="row-btn"
          onclick={() => toggle(u.id)}
          aria-expanded={expanded === u.id}
        >
          <!-- Thumbnail — 307-redirects to a presigned MinIO URL on the backend -->
          <img
            src={uploadImageUrl(u.id)}
            alt={u.filename}
            width="64"
            height="32"
            class="thumb"
          />

          <!-- Filename -->
          <span class="filename">{u.filename}</span>

          <!-- Human-readable local timestamp -->
          <span class="timestamp">{new Date(u.created_at).toLocaleString()}</span>
        </button>

        <!-- ── Expanded detail panel ──────────────────────────────────────── -->
        <!--
          Guarded by {#if expanded === u.id} so the log content (potentially long)
          is never inserted into the DOM until the user explicitly opens the row.
          Object.entries(u.results) gives us [scenarioId, resultFields] pairs in
          insertion order (m1, m2, m3, m4 as stored by the backend).
        -->
        {#if expanded === u.id}
          <div class="detail" role="region" aria-label="Detail for {u.filename}">
            {#each Object.entries(u.results) as [scenarioId, v]}
              <div class="scenario-block">
                <!-- Scenario human label (e.g. "M1 QLoRA") -->
                <h4 class="scenario-label">
                  {SCENARIO_LABEL[scenarioId] ?? scenarioId}
                </h4>

                <!-- Recognised text -->
                {#if v.text != null}
                  <p class="detail-text">{v.text}</p>
                {/if}

                <!-- Status badge (e.g. "Raw Output", "Corrected") -->
                {#if v.status_tag != null}
                  <span class="badge">{v.status_tag}</span>
                {/if}

                <!-- Metrics row: CER / WER / latency -->
                <dl class="metrics">
                  <dt>CER</dt>
                  <dd>{pct(v.cer)}</dd>
                  <dt>WER</dt>
                  <dd>{pct(v.wer)}</dd>
                  {#if v.latency_seconds != null}
                    <dt>Latency</dt>
                    <dd>{v.latency_seconds} s</dd>
                  {/if}
                </dl>

                <!-- Full model log / chain-of-thought -->
                {#if v.log != null}
                  <pre class="log">{v.log}</pre>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </li>
    {/each}
  </ul>
</div>

<style>
  /* ── Page wrapper ── */

  .history {
    max-width: 900px;
    margin: 0 auto;
    padding: 24px;
    font-family: 'Poppins', system-ui, sans-serif;
    color: #1f2937;
  }

  h2 {
    margin: 0 0 20px;
    font-size: 20px;
  }

  /* ── Error + empty messages ── */

  .error {
    color: #dc2626;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .empty {
    color: #64748b;
    background: #f8fafc;
    border: 1px solid #e1e6ef;
    border-radius: 8px;
    padding: 16px 20px;
  }

  /* ── Upload list ── */

  .upload-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .upload-item {
    background: #fff;
    border: 1px solid #e1e6ef;
    border-radius: 8px;
    overflow: hidden;
  }

  /* ── Row button (summary row) ── */

  .row-btn {
    /* Reset button defaults so it looks like a normal clickable row */
    all: unset;
    box-sizing: border-box;
    /* Flex layout: thumbnail | filename | timestamp */
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 10px 14px;
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
    color: #1f2937;
    /* Restore focus outline that `all: unset` removes — important for a11y */
    outline-offset: -2px;
  }

  .row-btn:hover {
    background: #f8fafc;
  }

  .row-btn:focus-visible {
    outline: 2px solid #4f46e5;
  }

  /* ── Thumbnail ── */

  .thumb {
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #e1e6ef;
    flex-shrink: 0;
    /* Neutral background while the image loads / 307-redirects */
    background: #f1f5f9;
  }

  /* ── Filename + timestamp ── */

  .filename {
    font-weight: 600;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .timestamp {
    color: #64748b;
    font-size: 12px;
    white-space: nowrap;
    flex-shrink: 0;
  }

  /* ── Expanded detail panel ── */

  .detail {
    border-top: 1px solid #e1e6ef;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  /* ── Per-scenario block inside the detail panel ── */

  .scenario-block {
    padding: 10px 12px;
    background: #f8fafc;
    border: 1px solid #e1e6ef;
    border-radius: 6px;
  }

  .scenario-label {
    margin: 0 0 6px;
    font-size: 13px;
    font-weight: 700;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .detail-text {
    margin: 0 0 6px;
    font-size: 14px;
    color: #1f2937;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ── Status badge ── */

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    background: #e0e7ff;
    color: #3730a3;
    margin-bottom: 8px;
  }

  /* ── Metrics definition list ── */

  .metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 16px;
    margin: 0 0 8px;
    font-size: 13px;
  }

  .metrics dt {
    font-weight: 600;
    color: #374151;
  }

  .metrics dd {
    margin: 0;
    color: #1f2937;
  }

  /* ── Model log / chain-of-thought ── */

  .log {
    margin: 0;
    padding: 8px 10px;
    background: #f1f5f9;
    border: 1px solid #e1e6ef;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
    color: #374151;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
  }
</style>
