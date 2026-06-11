import { parseNdjson } from './ndjson';
import type { DetectEvent, EvalRun, ScenarioSummary, UploadRecord } from './types';

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

// ─── SP-5 Dashboard / History API functions ──────────────────────────────────

/**
 * Fetch the list of all past batch-evaluation runs from `GET /v1/eval/runs`.
 *
 * Each element corresponds to one run of the full 4-scenario benchmark against
 * a named dataset.  The array is ordered newest-first by the backend.
 *
 * @param baseUrl - Backend root URL (defaults to `VITE_API_BASE` or localhost:8000).
 * @returns Array of {@link EvalRun} objects, possibly empty if no runs exist yet.
 * @throws {Error} On a non-OK HTTP response.
 */
export async function fetchEvalRuns(baseUrl: string = DEFAULT_BASE): Promise<EvalRun[]> {
  const res = await fetch(`${baseUrl}/v1/eval/runs`);
  if (!res.ok) throw new Error(`eval/runs failed: HTTP ${res.status}`);
  return (await res.json()) as EvalRun[];
}

/**
 * Fetch per-scenario aggregate metrics from `GET /v1/eval/summary`.
 *
 * When `runId` is supplied, the backend filters results to that specific run
 * (query param `?run_id=<id>`).  When `runId` is `null`, the endpoint returns
 * the summary for the most-recent run — useful for a "latest results" banner.
 *
 * @param runId  - The numeric eval-run ID to filter by, or `null` for the latest.
 * @param baseUrl - Backend root URL (defaults to `VITE_API_BASE` or localhost:8000).
 * @returns Array of {@link ScenarioSummary} rows (one per scenario, m1–m4).
 * @throws {Error} On a non-OK HTTP response.
 */
export async function fetchEvalSummary(
  runId: number | null,
  baseUrl: string = DEFAULT_BASE
): Promise<ScenarioSummary[]> {
  // Build optional query string — omit entirely when runId is null so the
  // backend interprets the request as "give me the most-recent run".
  const q = runId == null ? '' : `?run_id=${runId}`;
  const res = await fetch(`${baseUrl}/v1/eval/summary${q}`);
  if (!res.ok) throw new Error(`eval/summary failed: HTTP ${res.status}`);
  return (await res.json()) as ScenarioSummary[];
}

/**
 * Fetch a paginated list of user upload records from `GET /v1/uploads`.
 *
 * Used to populate the `/history` page table.  Each record includes the
 * original filename, S3 object key, optional ground truth, and a map of
 * per-scenario results stored in the `fold_results` JSONB column.
 *
 * @param limit   - Maximum number of rows to return (default 50).
 * @param offset  - Zero-based row offset for pagination (default 0).
 * @param baseUrl - Backend root URL (defaults to `VITE_API_BASE` or localhost:8000).
 * @returns Array of {@link UploadRecord} objects for the requested page.
 * @throws {Error} On a non-OK HTTP response.
 */
export async function fetchUploads(
  limit = 50,
  offset = 0,
  baseUrl: string = DEFAULT_BASE
): Promise<UploadRecord[]> {
  const res = await fetch(`${baseUrl}/v1/uploads?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`uploads failed: HTTP ${res.status}`);
  return (await res.json()) as UploadRecord[];
}

/**
 * Build the URL for fetching the raw image of a specific upload.
 *
 * Maps to `GET /v1/uploads/{id}/image` on the backend, which streams the
 * binary content of the stored image from S3/MinIO.  Intended for use in an
 * `<img src={uploadImageUrl(record.id)} />` binding inside the history page.
 *
 * This is a pure URL-builder (no network call) so it can be called
 * synchronously inside a template without `await`.
 *
 * @param uploadId - The numeric primary key of the upload row.
 * @param baseUrl  - Backend root URL (defaults to `VITE_API_BASE` or localhost:8000).
 * @returns The full URL string for the image endpoint.
 */
export function uploadImageUrl(uploadId: number, baseUrl: string = DEFAULT_BASE): string {
  return `${baseUrl}/v1/uploads/${uploadId}/image`;
}
