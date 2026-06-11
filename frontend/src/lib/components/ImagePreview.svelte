<script lang="ts">
  let { file }: { file: File | null } = $props();
  // Build/revoke an object URL whenever the file changes (no leak).
  let url = $state('');
  $effect(() => {
    if (!file) {
      url = '';
      return;
    }
    const u = URL.createObjectURL(file);
    url = u;
    return () => URL.revokeObjectURL(u);
  });
</script>

{#if url}
  <figure class="preview">
    <img src={url} alt="uploaded handwriting line" />
    <figcaption>{file?.name}</figcaption>
  </figure>
{/if}

<style>
  .preview { margin: 0; }
  .preview img { max-height: 90px; border: 1px solid #e1e6ef; border-radius: 6px; background: #fff; }
  figcaption { font-size: 12px; color: #94a3b8; margin-top: 4px; }
</style>
