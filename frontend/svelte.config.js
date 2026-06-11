import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // style: false — we use plain CSS only (no SCSS/Less/PostCSS), so we don't
  // need vite's CSS preprocessor. Disabling it avoids a vite version mismatch
  // between the project (vite 6) and vitest's bundled vite (5) that causes
  // "Cannot create proxy with a non-object" when running component tests.
  // NOTE: If you add SCSS/Less/PostCSS to any component, remove style:false first
  // (or align vitest's bundled vite with vite 6) — otherwise preprocessing is silently skipped.
  preprocess: vitePreprocess({ style: false }),
  kit: { adapter: adapter() }
};
export default config;
