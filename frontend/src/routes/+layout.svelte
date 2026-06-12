<script lang="ts">
  /**
   * Root layout — wraps every page in a persistent top navigation bar.
   *
   * Svelte 5 idiom used throughout:
   *   • `$props()`          — declares the `children` snippet prop (replaces `<slot/>`).
   *   • `{@render children()}` — renders the page content passed by SvelteKit.
   *   • `$app/stores` page  — reactive store that exposes the current URL so the
   *                           active link can be highlighted without JS-side routing logic.
   *
   * No `export let` statements are used; this file conforms to the Svelte 5 runes
   * convention followed by all SP-4 and SP-5 components in this project.
   */

  import { page } from '$app/stores';

  /** The page content injected by SvelteKit — replaces the legacy `<slot/>`. */
  let { children } = $props();

  /** Navigation links shown in the top bar. */
  const links = [
    { href: '/',          label: 'Detect'    },
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/history',   label: 'History'   }
  ];
</script>

<!--
  Persistent top navigation bar.
  The `class:active` directive compares the current pathname from the `$page`
  store against each link's href so the correct tab is underlined on every page.
-->
<nav aria-label="Main navigation">
  {#each links as link}
    <a
      href={link.href}
      class:active={$page.url.pathname === link.href}
      aria-current={$page.url.pathname === link.href ? 'page' : undefined}
    >
      {link.label}
    </a>
  {/each}
</nav>

<!-- Page content rendered below the nav bar. -->
<main>
  {@render children()}
</main>

<style>
  /* App-wide typeface. This lives in the layout (which wraps every route) so Poppins
     applies on every page, not just the home route. Components inherit this unless they
     deliberately override it (e.g. the monospace transcription/log areas). */
  :global(body) {
    font-family: 'Poppins', system-ui, sans-serif;
  }

  nav {
    display: flex;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #ddd;
    background: #fff;
    /* Inherit the global Poppins set on body. */
    font-family: inherit;
  }

  nav a {
    text-decoration: none;
    color: #444;
    font-weight: 600;
    padding-bottom: 2px;
  }

  nav a.active {
    color: #1a73e8;
    border-bottom: 2px solid #1a73e8;
  }

  main {
    padding: 1rem;
  }
</style>
