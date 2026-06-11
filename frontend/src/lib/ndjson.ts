/**
 * Incrementally parse a byte stream of newline-delimited JSON (NDJSON).
 *
 * The SP-2 /v1/detect endpoint streams one JSON object per line. We decode chunks as they
 * arrive, buffer any partial trailing line across chunk boundaries, and yield each complete
 * line as a parsed object — so the UI can render scenario columns the moment their event lands
 * rather than waiting for the whole response.
 */
export async function* parseNdjson(
  reader: ReadableStreamDefaultReader<Uint8Array>
): AsyncGenerator<unknown> {
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let nl: number;
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) yield JSON.parse(line);
    }
  }

  // Flush any final line that had no trailing newline.
  buffer += decoder.decode();
  const last = buffer.trim();
  if (last) yield JSON.parse(last);
}
