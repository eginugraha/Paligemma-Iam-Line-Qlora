<script lang="ts">
  import UploadArea from '$lib/components/UploadArea.svelte';
  import ImagePreview from '$lib/components/ImagePreview.svelte';
  import ScenarioColumn from '$lib/components/ScenarioColumn.svelte';
  import { detectStream } from '$lib/api';
  import type { DetectEvent, ResultEvent, ScenarioId } from '$lib/types';

  const SCENARIOS: { id: ScenarioId; title: string }[] = [
    { id: 'm1', title: 'Baseline QLoRA' },
    { id: 'm2', title: 'QLoRA + CoT' },
    { id: 'm3', title: 'QLoRA + RAG' },
    { id: 'm4', title: 'Hybrid CoT + RAG' }
  ];

  type ColState = 'idle' | 'pending' | 'filled' | 'error' | 'disabled';
  interface Col { state: ColState; result?: ResultEvent; errorMessage?: string }

  let file = $state<File | null>(null);
  let groundTruth = $state('');
  let running = $state(false);
  let topError = $state('');
  let hasGroundTruth = $state(false);
  let cols = $state<Record<ScenarioId, Col>>({
    m1: { state: 'idle' }, m2: { state: 'idle' }, m3: { state: 'idle' }, m4: { state: 'idle' }
  });

  function onfile(f: File) {
    file = f;
  }

  async function run() {
    if (!file || running) return;
    running = true;
    topError = '';
    hasGroundTruth = groundTruth.trim().length > 0;
    // All four columns start pending; any that never arrive become "disabled" at done.
    cols = { m1: { state: 'pending' }, m2: { state: 'pending' }, m3: { state: 'pending' }, m4: { state: 'pending' } };

    try {
      for await (const ev of detectStream(file, groundTruth.trim() || undefined)) {
        apply(ev);
      }
    } catch (e) {
      topError = e instanceof Error ? e.message : String(e);
      cols = { m1: { state: 'idle' }, m2: { state: 'idle' }, m3: { state: 'idle' }, m4: { state: 'idle' } };
    } finally {
      running = false;
    }
  }

  function apply(ev: DetectEvent) {
    if (ev.event === 'meta') {
      hasGroundTruth = ev.has_ground_truth;
    } else if (ev.event === 'result') {
      cols[ev.model] = { state: 'filled', result: ev };
    } else if (ev.event === 'error') {
      cols[ev.model] = { state: 'error', errorMessage: ev.message };
    } else if (ev.event === 'done') {
      // Any column still pending never streamed (RAG off / skipped) -> mark disabled.
      for (const { id } of SCENARIOS) {
        if (cols[id].state === 'pending') cols[id] = { state: 'disabled' };
      }
    }
  }
</script>

<main>
  <h1>HTR — Handwriting Recognition Compare</h1>
  <p class="sub">Upload one handwriting line; compare M1–M4 side by side.</p>

  <section class="controls">
    <UploadArea {onfile} disabled={running} />
    <div class="meta">
      <ImagePreview {file} />
      <label class="gt">
        Ground truth (optional)
        <input type="text" bind:value={groundTruth} placeholder="the quick brown fox" disabled={running} />
      </label>
      <button class="run" onclick={run} disabled={!file || running}>
        {running ? 'Running…' : '▶ Run M1–M4'}
      </button>
    </div>
  </section>

  {#if topError}<p class="top-error">⚠ {topError}</p>{/if}

  <section class="cols" aria-label="Scenario results">
    {#each SCENARIOS as s (s.id)}
      {#if cols[s.id].state !== 'idle'}
        <ScenarioColumn
          id={s.id}
          title={s.title}
          state={cols[s.id].state as 'pending' | 'filled' | 'error' | 'disabled'}
          result={cols[s.id].result}
          errorMessage={cols[s.id].errorMessage}
          {hasGroundTruth}
        />
      {/if}
    {/each}
  </section>
</main>

<style>
  :global(body) { margin: 0; font-family: 'Poppins', system-ui, sans-serif; background: #eef2f8; color: #1f2937; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 4px; font-size: 22px; }
  .sub { margin: 0 0 18px; color: #64748b; }
  .controls { display: flex; flex-direction: column; gap: 12px; background: #fff; border: 1px solid #e1e6ef; border-radius: 12px; padding: 16px; }
  .meta { display: flex; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
  .gt { display: flex; flex-direction: column; font-size: 12px; color: #64748b; gap: 4px; flex: 1; min-width: 200px; }
  .gt input { padding: 8px; border: 1px solid #cbd5e1; border-radius: 7px; font-size: 14px; }
  .run { background: #2563eb; color: #fff; border: 0; padding: 10px 18px; border-radius: 8px; font-weight: 700; cursor: pointer; }
  .run:disabled { opacity: 0.5; cursor: default; }
  .top-error { color: #dc2626; font-weight: 600; }
  .cols { display: flex; gap: 12px; margin-top: 18px; align-items: stretch; }
</style>
