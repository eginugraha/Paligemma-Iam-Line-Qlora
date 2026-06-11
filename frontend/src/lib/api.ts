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
