<script lang="ts">
  let {
    onfile,
    disabled = false
  }: { onfile: (file: File) => void; disabled?: boolean } = $props();

  let error = $state('');
  const OK = ['image/png', 'image/jpeg'];

  function accept(file: File | undefined) {
    if (!file) return;
    if (!OK.includes(file.type)) {
      error = 'Please choose a png, jpg, or jpeg image.';
      return;
    }
    error = '';
    onfile(file);
  }

  function onChange(e: Event) {
    accept((e.currentTarget as HTMLInputElement).files?.[0]);
  }
  function onDrop(e: DragEvent) {
    e.preventDefault();
    if (disabled) return;
    accept(e.dataTransfer?.files?.[0]);
  }
</script>

<div
  class="drop"
  class:disabled
  role="button"
  tabindex="0"
  ondragover={(e) => { if (!disabled) e.preventDefault(); }}
  ondrop={onDrop}
>
  <p>⬆ Drag &amp; drop a handwriting-line image here, or</p>
  <label class="btn">
    Choose file
    <input
      data-testid="file-input"
      type="file"
      accept="image/png,image/jpeg"
      {disabled}
      onchange={onChange}
      hidden
    />
  </label>
  {#if error}<p class="error">{error}</p>{/if}
</div>

<style>
  .drop { border: 2px dashed #9aa7bd; border-radius: 10px; padding: 22px; text-align: center; background: #f6f8fc; color: #5b6b86; }
  .drop.disabled { opacity: 0.6; }
  .btn { display: inline-block; margin-top: 8px; background: #2563eb; color: #fff; padding: 7px 14px; border-radius: 7px; font-weight: 600; cursor: pointer; }
  .error { color: #dc2626; margin-top: 8px; font-size: 13px; }
</style>
