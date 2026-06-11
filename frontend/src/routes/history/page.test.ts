/**
 * Test suite for /history (+page.svelte)
 *
 * Strategy:
 *   - $lib/api is mocked so the test never hits a real backend.
 *     `fetchUploads` returns one canned UploadRecord; `uploadImageUrl` is a pure
 *     URL-builder stub.
 *   - The test asserts:
 *       1. The filename appears in the DOM after the async $effect resolves.
 *       2. The thumbnail <img> has the correct src URL.
 *       3. The expanded detail (log text) is NOT in the DOM until the row is clicked.
 *       4. After clicking the row, the log text IS visible (toggle expanded).
 *
 * Mock resolution:
 *   Vitest resolves `$lib/...` via the SvelteKit vite plugin alias ($lib → src/lib).
 *   vi.mock() factories are hoisted above all imports by Vite, so the alias is
 *   already resolved when mocks are registered — same pattern as dashboard/page.test.ts.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';

// ─── Mock $lib/api ─────────────────────────────────────────────────────────────
// fetchUploads returns one upload record whose result map has an m1 entry.
// uploadImageUrl is a pure function — we stub it to return a predictable URL so
// the <img src={...}> assertion works without a live backend.
vi.mock('$lib/api', () => ({
  fetchUploads: vi.fn(async () => [
    {
      id: 1,
      created_at: '2026-06-11T00:00:00Z',
      filename: 'a.png',
      object_key: 'uploads/a.png',
      ground_truth: 'the cat',
      results: {
        m1: {
          text: 'the cat',
          cer: 0,
          wer: 0,
          latency_seconds: 0.7,
          log: 'Direct.',
          status_tag: 'Raw Output'
        }
      }
    }
  ]),
  // Pure URL-builder stub — returns a predictable URL without the DEFAULT_BASE
  // environment variable, which is undefined in the vitest jsdom environment.
  uploadImageUrl: (id: number) => `http://api/v1/uploads/${id}/image`
}));

// Import AFTER mock declarations (hoisting guarantees mock wins anyway, but
// keeping the import below is conventional and consistent with dashboard tests).
import History from './+page.svelte';

afterEach(() => vi.clearAllMocks());

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe('/history', () => {
  it('lists uploads with a thumbnail and expands detail on click', async () => {
    render(History);

    // ── Wait for the async $effect (fetchUploads) to resolve ──────────────────
    // The filename appears in the list once the effect resolves and Svelte flushes.
    await waitFor(() => expect(screen.getByText('a.png')).toBeInTheDocument());

    // ── Thumbnail src matches uploadImageUrl(1) ────────────────────────────────
    // getByRole('img') is safe here because there is exactly one upload row so
    // exactly one thumbnail <img> is in the DOM.
    const img = screen.getByRole('img') as HTMLImageElement;
    expect(img.src).toBe('http://api/v1/uploads/1/image');

    // ── Log text must NOT be in the DOM before the row is expanded ────────────
    // Verifies the {#if expanded === u.id} guard keeps the detail panel hidden.
    expect(screen.queryByText('Direct.')).not.toBeInTheDocument();

    // ── Click the row to expand the detail panel ──────────────────────────────
    // The filename text is inside the <button> for the row; fireEvent.click bubbles
    // up to the button's onclick handler, triggering the toggle(u.id) call.
    await fireEvent.click(screen.getByText('a.png'));

    // ── Log text must now be visible ──────────────────────────────────────────
    expect(screen.getByText('Direct.')).toBeInTheDocument();
  });
});
