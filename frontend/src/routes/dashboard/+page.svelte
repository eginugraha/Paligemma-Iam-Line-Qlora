<script lang="ts">
  /**
   * /dashboard — FR-FE-05: Batch-Evaluation Statistics Dashboard
   *
   * Shows a comparison matrix (table) and a grouped bar chart comparing average CER,
   * WER, and latency across the four HTR scenarios (M1–M4) for one evaluation run.
   *
   * Data flow:
   *   1. On mount, `fetchEvalRuns()` loads the list of past batch-eval runs
   *      (newest-first from the backend).
   *   2. The newest run (index 0) is selected by default.
   *   3. `fetchEvalSummary(selectedRunId)` loads per-scenario aggregates for that run.
   *   4. The user can pick a different run via the <select> dropdown; changing it
   *      re-invokes `fetchEvalSummary` reactively.
   *
   * Svelte 5 runes used:
   *   $state    — mutable reactive variables (runs, selectedRunId, summary, error).
   *   $derived  — derived bar-chart labels and datasets from summary (no explicit stores).
   *   $effect   — initial data load on mount (replaces onMount + reactive statements).
   */

  import { fetchEvalRuns, fetchEvalSummary } from '$lib/api';
  import type { EvalRun, ScenarioSummary, ScenarioId } from '$lib/types';
  import BarChart from '$lib/BarChart.svelte';

  // ─── Scenario label map ───────────────────────────────────────────────────────

  /**
   * Maps each internal scenario id to a human-readable label for the table header
   * and bar-chart x-axis.  Matches the thesis Bab 4 terminology.
   */
  const SCENARIO_LABEL: Record<ScenarioId, string> = {
    m1: 'M1 QLoRA',
    m2: 'M2 +CoT',
    m3: 'M3 +RAG',
    m4: 'M4 Hybrid'
  };

  // ─── State ────────────────────────────────────────────────────────────────────

  /** Full list of evaluation runs returned by the backend (newest-first). */
  let runs = $state<EvalRun[]>([]);

  /**
   * The currently selected run id, or `null` before any run is loaded.
   * `null` is also passed to `fetchEvalSummary` to request the most-recent run,
   * but in practice we always assign the newest run id from `runs[0]` first.
   */
  let selectedRunId = $state<number | null>(null);

  /** Per-scenario aggregate metrics for the selected run. */
  let summary = $state<ScenarioSummary[]>([]);

  /** Non-empty string when a fetch fails; displayed as an error banner. */
  let error = $state('');

  // ─── Derived chart data ───────────────────────────────────────────────────────

  /**
   * X-axis labels for the bar chart — one per summary row, mapped from scenario id
   * to display name.  Re-derived whenever `summary` changes.
   */
  let labels = $derived(
    summary.map((s) => SCENARIO_LABEL[s.scenario] ?? s.scenario)
  );

  /**
   * Two dataset series (Avg CER and Avg WER) consumed by the BarChart component.
   * Null metric values are coerced to 0 so Chart.js never receives null.
   */
  let datasets = $derived([
    { label: 'Avg CER', data: summary.map((s) => s.avg_cer ?? 0) },
    { label: 'Avg WER', data: summary.map((s) => s.avg_wer ?? 0) }
  ]);

  // ─── Data loading ─────────────────────────────────────────────────────────────

  /**
   * Fetch the per-scenario summary for the currently selected run and update state.
   * Called both on initial mount (after runs load) and when the selector changes.
   */
  async function loadSummary() {
    try {
      summary = await fetchEvalSummary(selectedRunId);
    } catch (e) {
      // Clear any stale rows from the previous run so an error banner is never
      // shown alongside data that no longer matches the selected run.
      summary = [];
      error = e instanceof Error ? e.message : String(e);
    }
  }

  /**
   * Initial data load — runs once when the component mounts (Svelte 5 `$effect`
   * with no reactive reads after the first tick, so it does not re-run).
   *
   * Sequence:
   *   1. Load the run list.
   *   2. Select the newest run (index 0, since the backend returns newest-first).
   *   3. Load that run's scenario summary.
   *
   * If there are no runs, the "no runs" message is shown and we skip step 3.
   */
  $effect(() => {
    // Svelte 5 tracks reactive reads that happen SYNCHRONOUSLY in the $effect
    // body before the first `await`.  Because we immediately hand off to an async
    // IIFE, *all* reactive reads happen after the first await boundary — Svelte
    // never sees them and therefore never registers a dependency.  That is why
    // this block runs exactly once (on mount) rather than re-running whenever
    // `runs`, `selectedRunId`, `summary`, or `error` change.
    // The async IIFE is needed because $effect callbacks must be synchronous
    // themselves (they may return a cleanup function, not a Promise).
    (async () => {
      try {
        runs = await fetchEvalRuns();
        if (runs.length > 0) {
          selectedRunId = runs[0].id;   // newest-first → index 0 is the latest
          await loadSummary();
        }
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
      }
    })();
  });

  // ─── Formatting helpers ───────────────────────────────────────────────────────

  /**
   * Format a nullable percentage value for the matrix table.
   * `null` (no ground truth) is represented as an em-dash per thesis convention.
   */
  function pct(v: number | null): string {
    return v == null ? '—' : `${v}%`;
  }

  /**
   * Format a nullable latency value (seconds) for the matrix table.
   * `null` (timing not recorded) is represented as an em-dash.
   */
  function secs(v: number | null): string {
    return v == null ? '—' : `${v} s`;
  }
</script>

<!--
  Dashboard page — two sections:
    1. Run selector + scenario matrix table.
    2. Bar chart (CER vs WER grouped by scenario).
-->

<div class="dashboard">
  <h2>Batch Evaluation Dashboard</h2>

  <!-- ── Error banner ─────────────────────────────────────────────────────── -->
  {#if error}
    <p class="error" role="alert">⚠ {error}</p>
  {/if}

  <!-- ── No-runs message ──────────────────────────────────────────────────── -->
  {#if runs.length === 0 && !error}
    <p class="no-runs">
      No evaluation runs found. Run
      <code>python scripts/eval_sp5.py</code>
      first to generate benchmark data.
    </p>
  {/if}

  <!-- ── Run selector ─────────────────────────────────────────────────────── -->
  {#if runs.length > 0}
    <div class="selector-row">
      <label for="run-select">Evaluation run:</label>
      <!-- Svelte 5 event syntax: onchange instead of on:change -->
      <select
        id="run-select"
        bind:value={selectedRunId}
        onchange={loadSummary}
      >
        {#each runs as run (run.id)}
          <option value={run.id}>
            #{run.id} — {run.dataset} ({run.n_samples} samples) · {run.created_at.slice(0, 10)}
          </option>
        {/each}
      </select>
    </div>

    <!-- ── Scenario matrix table ──────────────────────────────────────────── -->
    <!--
      One row per ScenarioSummary returned by the backend (m1–m4 in the order
      the backend sends them).  avg_cer and avg_wer are percentage values already
      (0–100), so we append "%" directly.  avg_latency_seconds is in seconds.
    -->
    <table class="matrix" aria-label="Scenario comparison matrix">
      <thead>
        <tr>
          <th scope="col">Scenario</th>
          <th scope="col">Avg CER</th>
          <th scope="col">Avg WER</th>
          <th scope="col">Avg Latency</th>
          <th scope="col">N</th>
        </tr>
      </thead>
      <tbody>
        {#each summary as row (row.scenario)}
          <tr>
            <td>{SCENARIO_LABEL[row.scenario] ?? row.scenario}</td>
            <td>{pct(row.avg_cer)}</td>
            <td>{pct(row.avg_wer)}</td>
            <td>{secs(row.avg_latency_seconds)}</td>
            <td>{row.n}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    <!-- ── Comparison bar chart ───────────────────────────────────────────── -->
    <!--
      Chart.js grouped bar chart showing Avg CER and Avg WER per scenario.
      The `labels` and `datasets` derived values auto-update when `summary` changes.
      BarChart internally uses $effect + bind:this on a <canvas> element.
    -->
    {#if summary.length > 0}
      <div class="chart-container">
        <BarChart {labels} {datasets} />
      </div>
    {/if}
  {/if}
</div>

<style>
  .dashboard {
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

  /* ── Error + no-runs messages ── */

  .error {
    color: #dc2626;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .no-runs {
    color: #64748b;
    background: #f8fafc;
    border: 1px solid #e1e6ef;
    border-radius: 8px;
    padding: 16px 20px;
  }

  .no-runs code {
    font-family: monospace;
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
  }

  /* ── Run selector ── */

  .selector-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    font-size: 14px;
  }

  .selector-row label {
    font-weight: 600;
    color: #374151;
  }

  .selector-row select {
    padding: 6px 10px;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    font-size: 14px;
    background: #fff;
    cursor: pointer;
  }

  /* ── Scenario matrix table ── */

  .matrix {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin-bottom: 28px;
    background: #fff;
    border: 1px solid #e1e6ef;
    border-radius: 8px;
    overflow: hidden;
  }

  .matrix th,
  .matrix td {
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #e1e6ef;
  }

  .matrix thead tr {
    background: #f1f5f9;
  }

  .matrix th {
    font-weight: 700;
    color: #374151;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .matrix tbody tr:last-child td {
    border-bottom: none;
  }

  .matrix tbody tr:hover {
    background: #f8fafc;
  }

  /* ── Bar chart ── */

  .chart-container {
    background: #fff;
    border: 1px solid #e1e6ef;
    border-radius: 8px;
    padding: 20px;
  }
</style>
