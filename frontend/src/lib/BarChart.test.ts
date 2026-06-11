import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/svelte';

// vi.mock factories are hoisted above imports, so any variable they reference must also be
// hoisted. vi.hoisted() returns values created in that same hoisted phase, avoiding the
// Temporal-Dead-Zone ReferenceError you get when a factory closes over a plain top-level const.
const { ChartMock, destroy } = vi.hoisted(() => {
  const destroy = vi.fn();
  // Type the constructor params (canvas, config) so `ChartMock.mock.calls[0][1]` is a typed
  // tuple element under svelte-check — without this, vi.fn() infers a zero-length args tuple
  // and indexing `[1]` is a type error.
  const ChartMock = vi.fn((_canvas: unknown, _config: { type: string; data: { labels: string[]; datasets: { label: string; data: number[] }[] } }) => ({
    destroy,
    update: vi.fn(),
    data: { labels: [], datasets: [] }
  }));
  return { ChartMock, destroy };
});
vi.mock('chart.js/auto', () => ({ default: ChartMock }));

import BarChart from './BarChart.svelte';

// Clear BEFORE each test so a prior test's @testing-library auto-cleanup (which unmounts and
// calls the mocked destroy) cannot leak call counts into the next test.
beforeEach(() => vi.clearAllMocks());

describe('BarChart.svelte', () => {
  const labels = ['M1', 'M2', 'M3', 'M4'];
  const datasets = [
    { label: 'Avg CER', data: [17, 16, 5, 5] },
    { label: 'Avg WER', data: [28, 27, 10, 9] }
  ];

  it('constructs a Chart with the passed labels and datasets', () => {
    render(BarChart, { props: { labels, datasets } });
    expect(ChartMock).toHaveBeenCalledTimes(1);
    const cfg = ChartMock.mock.calls[0][1];
    expect(cfg.type).toBe('bar');
    expect(cfg.data.labels).toEqual(labels);
    expect(cfg.data.datasets[0].label).toBe('Avg CER');
  });

  it('destroys the chart on unmount', () => {
    const { unmount } = render(BarChart, { props: { labels, datasets } });
    unmount();
    expect(destroy).toHaveBeenCalledTimes(1);
  });
});
