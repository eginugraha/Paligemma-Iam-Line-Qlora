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
