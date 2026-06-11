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

// ─── SP-5 Dashboard / History types ─────────────────────────────────────────

/**
 * A batch-evaluation run — one row of the `eval_run` table created whenever the
 * user triggers a full 4-scenario benchmark on a named dataset.
 *
 * Returned by `GET /v1/eval/runs` and used to populate the run selector on the
 * dashboard page.
 */
export interface EvalRun {
  /** Auto-increment primary key. */
  id: number;
  /** ISO-8601 UTC timestamp when the run was created. */
  created_at: string;
  /** Human-readable name for the dataset that was benchmarked (e.g. "iam-line-test"). */
  dataset: string;
  /** Number of image samples included in this run. */
  n_samples: number;
  /**
   * Optional reference to the fine-tuned model checkpoint that was active for
   * this run (e.g. a RunPod artifact path). `null` if no custom checkpoint was
   * specified and the base model was used.
   */
  model_ref: string | null;
  /** Whether the RAG/pgvector corrector (M3/M4) was enabled for this run. */
  rag_enabled: boolean;
}

/**
 * Per-scenario aggregate metrics for one eval run — one row of the dashboard
 * comparison matrix.
 *
 * Returned by `GET /v1/eval/summary?run_id={id}` (or without the query param
 * for the most-recent run).  One `ScenarioSummary` exists for each of the four
 * scenarios (m1–m4) in a run.
 */
export interface ScenarioSummary {
  /** Which of the four HTR scenarios these metrics describe. */
  scenario: ScenarioId;
  /**
   * Mean Character Error Rate across all samples in the run for this scenario,
   * expressed as a percentage (0–100).  `null` when no ground-truth labels were
   * available for any sample.
   */
  avg_cer: number | null;
  /**
   * Mean Word Error Rate, same conditions as `avg_cer`.  `null` without ground
   * truth.
   */
  avg_wer: number | null;
  /**
   * Mean end-to-end inference latency in seconds, measured server-side.  `null`
   * if timing was not recorded for this run.
   */
  avg_latency_seconds: number | null;
  /** Number of samples that contributed to these averages. */
  n: number;
}

/**
 * One end-user upload history record — mirrors the `upload` DB row plus its
 * denormalised per-scenario results.
 *
 * Returned by `GET /v1/uploads?limit=&offset=` for the history page.
 */
export interface UploadRecord {
  /** Auto-increment primary key of the upload row. */
  id: number;
  /** ISO-8601 UTC timestamp when the image was uploaded. */
  created_at: string;
  /** Original filename supplied by the browser (e.g. "page_001.png"). */
  filename: string;
  /** S3 / MinIO object key where the image is stored (e.g. "uploads/abc.png"). */
  object_key: string;
  /**
   * Optional reference transcription the user supplied at upload time.  `null`
   * when no ground truth was provided, in which case CER/WER fields will also be
   * `null`.
   */
  ground_truth: string | null;
  /**
   * Map of scenario id → partial result fields, stored as the `fold_results`
   * JSONB blob in the database.  Each value may be absent or partially populated
   * depending on which scenarios completed successfully.
   *
   * Example shape:
   * ```json
   * { "m1": { "text": "Hello world", "cer": 0.05, "latency_seconds": 1.2 } }
   * ```
   */
  results: Record<
    string,
    {
      /** Recognised text produced by this scenario. */
      text?: string;
      /** Character Error Rate (percentage, 0–100).  `null` without ground truth. */
      cer?: number | null;
      /** Word Error Rate (percentage, 0–100).  `null` without ground truth. */
      wer?: number | null;
      /** Server-side inference latency in seconds.  `null` if not recorded. */
      latency_seconds?: number | null;
      /** Raw reasoning / chain-of-thought log from the model. */
      log?: string;
      /** Human-readable processing stage label, e.g. "Corrected" or "Raw Output". */
      status_tag?: string;
    }
  >;
}
