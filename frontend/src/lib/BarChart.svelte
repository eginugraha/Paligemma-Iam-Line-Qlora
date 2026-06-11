<script lang="ts">
  /**
   * BarChart — a thin Svelte 5 wrapper around Chart.js (the one intentional UI-library
   * exception in this project, used for the thesis Bab 4 comparison visual).
   *
   * `chart.js/auto` is statically imported: it auto-registers the bar controller, scales,
   * and elements so we never hand-register Chart.js components. The chart is constructed
   * inside a `$effect` once the <canvas> is bound, and destroyed on unmount (or before the
   * effect re-runs) to avoid leaking canvas listeners. Because `labels`/`datasets` are read
   * inside the effect, Svelte rebuilds the chart whenever the parent passes new data.
   */
  import Chart from 'chart.js/auto';

  let {
    labels = [],
    datasets = []
  }: {
    /** X-axis tick labels, one per bar group, e.g. ['M1','M2','M3','M4']. */
    labels: string[];
    /** Chart.js dataset descriptors — a label + one numeric value per scenario. */
    datasets: { label: string; data: number[] }[];
  } = $props();

  /** DOM reference to the <canvas>, written by bind:this once the element mounts. */
  let canvas: HTMLCanvasElement | undefined = $state(undefined);

  /** The live Chart.js instance; undefined until the first build. */
  let chart: Chart | undefined;

  $effect(() => {
    // Wait until the canvas element is bound. Reading `labels`/`datasets` here also registers
    // them as reactive dependencies so the chart rebuilds when the parent passes new data.
    if (!canvas) return;

    // Tear down any previous chart (handles prop-change re-runs).
    chart?.destroy();

    chart = new Chart(canvas, {
      type: 'bar',
      data: { labels, datasets },
      options: { responsive: true, scales: { y: { beginAtZero: true } } }
    });

    // Cleanup runs before the next effect re-run and on unmount. Null out first so a second
    // invocation is a no-op via optional chaining.
    return () => {
      const c = chart;
      chart = undefined;
      c?.destroy();
    };
  });
</script>

<canvas bind:this={canvas} aria-label="Scenario comparison bar chart"></canvas>
