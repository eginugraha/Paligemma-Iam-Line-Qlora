<script lang="ts">
  import type { ResultEvent } from '$lib/types';

  let {
    id,
    title,
    state,
    result,
    errorMessage = '',
    hasGroundTruth = false
  }: {
    id: string;
    title: string;
    state: 'pending' | 'filled' | 'error' | 'disabled';
    result?: ResultEvent;
    errorMessage?: string;
    hasGroundTruth?: boolean;
  } = $props();

  // status_tag -> badge colour class (one fixed mapping for the four scenarios).
  const BADGE: Record<string, string> = {
    'Raw Output': 'b1',
    Reasoned: 'b2',
    Corrected: 'b3',
    Optimal: 'b4'
  };
  const badgeClass = $derived(result ? (BADGE[result.status_tag] ?? 'b1') : 'b1');
  const metric = (v: number | null) => (v === null || !hasGroundTruth ? '—' : v.toFixed(2));
</script>

<div class="col" class:err={state === 'error'} class:dis={state === 'disabled'}>
  <div class="head">
    <span class="id">{id.toUpperCase()}</span>
    <span class="title">{title}</span>
  </div>

  {#if state === 'pending'}
    <div class="spinner" data-testid="spinner" role="status" aria-label="loading"></div>
    <p class="muted center">running…</p>
  {:else if state === 'filled' && result}
    <span class="badge {badgeClass}">{result.status_tag}</span>
    <p class="text">{result.text}</p>
    <p class="row"><b>CER</b> <span>{metric(result.cer)}</span> &nbsp; <b>WER</b> <span>{metric(result.wer)}</span></p>
    <p class="row">⏱ {result.latency_seconds.toFixed(2)}s</p>
    <pre class="log">{result.log}</pre>
  {:else if state === 'error'}
    <span class="badge berr">Failed</span>
    <p class="text muted">⚠ {errorMessage}</p>
  {:else}
    <p class="muted center">not run — enable RAG (HTR_ENABLE_RAG=1)</p>
  {/if}
</div>

<style>
  .col {
    flex: 1;
    min-width: 0;
    background: #fff;
    border: 1px solid #e1e6ef;
    border-radius: 9px;
    padding: 12px;
  }
  .col.err {
    border-color: #f0c9c9;
    background: #fdf6f6;
  }
  .col.dis {
    background: #f7f9fc;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 8px;
  }
  .id {
    font-weight: 800;
    color: #111827;
  }
  .title {
    font-size: 12px;
    color: #94a3b8;
  }
  .badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    color: #fff;
  }
  .b1 {
    background: #6b7280;
  }
  .b2 {
    background: #7c3aed;
  }
  .b3 {
    background: #2563eb;
  }
  .b4 {
    background: #16a34a;
  }
  .berr {
    background: #dc2626;
  }
  .text {
    font-weight: 700;
    font-size: 15px;
    color: #1f2937;
    margin: 8px 0;
    overflow-wrap: anywhere;
  }
  .row {
    font-size: 12px;
    color: #475569;
    margin: 3px 0;
  }
  .row b {
    color: #111827;
  }
  .log {
    font-size: 11px;
    color: #475569;
    background: #eef2f8;
    border-radius: 6px;
    padding: 8px;
    margin-top: 8px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-family: ui-monospace, monospace;
  }
  .muted {
    color: #94a3b8;
  }
  .center {
    text-align: center;
  }
  .spinner {
    width: 22px;
    height: 22px;
    border: 3px solid #d7deea;
    border-top-color: #2563eb;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 14px auto;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
