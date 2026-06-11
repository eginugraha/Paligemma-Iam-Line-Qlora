import { describe, it, expect, vi, afterEach } from 'vitest';
import { fetchEvalRuns, fetchEvalSummary, fetchUploads, uploadImageUrl } from './api';

afterEach(() => vi.restoreAllMocks());

function mockFetchJson(payload: unknown) {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => payload }) as Response));
}

describe('dashboard/history api', () => {
  it('fetchEvalRuns returns parsed runs', async () => {
    mockFetchJson([{ id: 7, dataset: 'iam-line-test', n_samples: 2, model_ref: 'x', rag_enabled: true,
      created_at: '2026-06-11T00:00:00Z' }]);
    const runs = await fetchEvalRuns('http://api');
    expect(runs[0].id).toBe(7);
    expect(fetch).toHaveBeenCalledWith('http://api/v1/eval/runs');
  });

  it('fetchEvalSummary passes run_id when given', async () => {
    mockFetchJson([{ scenario: 'm1', avg_cer: 5, avg_wer: 10, avg_latency_seconds: 0.7, n: 2 }]);
    const s = await fetchEvalSummary(7, 'http://api');
    expect(s[0].scenario).toBe('m1');
    expect(fetch).toHaveBeenCalledWith('http://api/v1/eval/summary?run_id=7');
  });

  it('fetchEvalSummary omits run_id when null', async () => {
    mockFetchJson([]);
    await fetchEvalSummary(null, 'http://api');
    expect(fetch).toHaveBeenCalledWith('http://api/v1/eval/summary');
  });

  it('fetchUploads builds pagination query', async () => {
    mockFetchJson([{ id: 1, filename: 'a.png', object_key: 'uploads/a.png', ground_truth: null,
      results: {}, created_at: '2026-06-11T00:00:00Z' }]);
    const u = await fetchUploads(20, 0, 'http://api');
    expect(u[0].filename).toBe('a.png');
    expect(fetch).toHaveBeenCalledWith('http://api/v1/uploads?limit=20&offset=0');
  });

  it('uploadImageUrl builds the image endpoint URL', () => {
    expect(uploadImageUrl(1, 'http://api')).toBe('http://api/v1/uploads/1/image');
  });
});
