/**
 * Test suite for /dashboard (+page.svelte)
 *
 * Strategy:
 *   - The BarChart component is stubbed out (replaced with an empty <div>) so the
 *     test does not pull in chart.js or require a real <canvas>.
 *   - The $lib/api module is mocked with vi.fn() returning canned data; this lets
 *     us assert on DOM output without a live backend.
 *
 * Mock resolution:
 *   Vitest resolves `$lib/...` aliases via the SvelteKit vite plugin (sveltekit()
 *   in vite.config.ts), which maps $lib → src/lib.  vi.mock() factories are
 *   hoisted above all imports by Vite's module graph, so the alias is already
 *   resolved when our mocks are registered — `$lib/api` and `$lib/BarChart.svelte`
 *   both work without needing relative-path equivalents.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/svelte';

// ─── Mock BarChart ─────────────────────────────────────────────────────────────
// Replace the real BarChart (which needs chart.js + a canvas environment) with a
// minimal stub that renders a single <div>.  vi.mock is hoisted so this takes
// effect before the Dashboard component is imported below.
vi.mock('$lib/BarChart.svelte', async () => {
  // Dynamic import inside the factory so we get the Svelte-compiled stub class.
  const Stub = (await import('../../lib/__stubs__/Empty.svelte')).default;
  return { default: Stub };
});

// ─── Mock API ─────────────────────────────────────────────────────────────────
// Default factory: returns a single eval run (id=7) and two scenario summaries
// so we can assert on formatted cell values, label map rendering, and full row
// output.  Individual tests may override the return value per-call using
// vi.mocked(...).mockResolvedValueOnce(...) — those overrides win for one
// invocation and then the factory default resumes.
vi.mock('$lib/api', () => ({
  fetchEvalRuns: vi.fn(async () => [
    {
      id: 7,
      created_at: '2026-06-11T00:00:00Z',
      dataset: 'iam-line-test',
      n_samples: 2,
      model_ref: 'x',
      rag_enabled: true
    }
  ]),
  fetchEvalSummary: vi.fn(async () => [
    // M1 scenario — has full metrics
    { scenario: 'm1', avg_cer: 17.4, avg_wer: 28.3, avg_latency_seconds: 0.78, n: 2 },
    // M3 scenario — avg_cer is 5.0, which JS renders as "5%" (no trailing .0)
    { scenario: 'm3', avg_cer: 5.0, avg_wer: 10.0, avg_latency_seconds: 1.1, n: 2 }
  ])
}));

// Import AFTER mocks are declared (hoisting guarantees mocks win anyway, but
// keeping the import below is conventional and easier to read).
import Dashboard from './+page.svelte';
import { fetchEvalRuns } from '$lib/api';

afterEach(() => vi.clearAllMocks());

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe('/dashboard', () => {
  it('renders one matrix row per scenario from the summary', async () => {
    render(Dashboard);

    // Wait for the async $effect (fetchEvalRuns → fetchEvalSummary) to complete
    // and Svelte to flush the DOM update.
    await waitFor(() => expect(screen.getByText('17.4%')).toBeInTheDocument());

    // ── M1 row: metrics + label from SCENARIO_LABEL map ──────────────────────
    // Confirm the label map is applied correctly (internal id "m1" → "M1 QLoRA").
    expect(screen.getByText('M1 QLoRA')).toBeInTheDocument();
    expect(screen.getByText('28.3%')).toBeInTheDocument();
    expect(screen.getByText('0.78 s')).toBeInTheDocument();

    // ── M3 row: full row rendering including label, N value, and latency ─────
    // "M3 +RAG" proves the label map covers m3.
    expect(screen.getByText('M3 +RAG')).toBeInTheDocument();
    // 5.0 formatted with `${v}%` → "5%" (JS drops the trailing .0)
    expect(screen.getByText('5%')).toBeInTheDocument();
    // N value for M3 — confirms the `n` field is rendered for full rows.
    // Both rows have n=2, so getAllByText is needed to avoid a "found multiple" error.
    expect(screen.getAllByText('2')).toHaveLength(2);
    // M3 latency in seconds.
    expect(screen.getByText('1.1 s')).toBeInTheDocument();
  });

  it('shows the no-runs message and the eval script hint when fetchEvalRuns returns []', async () => {
    // Override fetchEvalRuns to return an empty list for this test only.
    // vi.mocked() gives us a typed reference to the already-mocked function;
    // mockResolvedValueOnce replaces the return value for a single invocation and
    // then the factory default resumes (though afterEach → clearAllMocks resets it).
    vi.mocked(fetchEvalRuns).mockResolvedValueOnce([]);

    render(Dashboard);

    // The no-runs branch should appear once the effect resolves.
    await waitFor(() =>
      expect(screen.getByText(/python scripts\/eval_sp5\.py/i)).toBeInTheDocument()
    );

    // The selector and table must NOT render — there are no runs to display.
    expect(screen.queryByRole('combobox')).toBeNull();
    expect(screen.queryByRole('table')).toBeNull();
  });
});
