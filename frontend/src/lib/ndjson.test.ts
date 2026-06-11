import { describe, it, expect } from 'vitest';
import { parseNdjson } from './ndjson';

// Build a ReadableStreamDefaultReader-like object from string chunks.
function readerFrom(chunks: string[]): ReadableStreamDefaultReader<Uint8Array> {
  const enc = new TextEncoder();
  let i = 0;
  return {
    read: async () =>
      i < chunks.length
        ? { done: false, value: enc.encode(chunks[i++]) }
        : { done: true, value: undefined }
  } as unknown as ReadableStreamDefaultReader<Uint8Array>;
}

// Build a ReadableStreamDefaultReader-like object from raw Uint8Array chunks.
// Used to test multi-byte UTF-8 characters split across chunk boundaries.
function readerFromBytes(chunks: Uint8Array[]): ReadableStreamDefaultReader<Uint8Array> {
  let i = 0;
  return {
    read: async () =>
      i < chunks.length
        ? { done: false, value: chunks[i++] }
        : { done: true, value: undefined }
  } as unknown as ReadableStreamDefaultReader<Uint8Array>;
}

async function collect(chunks: string[]): Promise<unknown[]> {
  const out: unknown[] = [];
  for await (const obj of parseNdjson(readerFrom(chunks))) out.push(obj);
  return out;
}

describe('parseNdjson', () => {
  it('parses one object per line', async () => {
    const out = await collect(['{"a":1}\n{"a":2}\n']);
    expect(out).toEqual([{ a: 1 }, { a: 2 }]);
  });

  it('joins a line split across chunks', async () => {
    const out = await collect(['{"ev":"me', 'ta"}\n']);
    expect(out).toEqual([{ ev: 'meta' }]);
  });

  it('handles multiple lines in one chunk and a missing trailing newline', async () => {
    const out = await collect(['{"a":1}\n{"a":2}', '\n{"a":3}']);
    expect(out).toEqual([{ a: 1 }, { a: 2 }, { a: 3 }]);
  });

  it('ignores blank lines', async () => {
    const out = await collect(['\n{"a":1}\n\n']);
    expect(out).toEqual([{ a: 1 }]);
  });

  it('handles a multi-byte UTF-8 character split across chunks', async () => {
    // 'é' is bytes 0xC3 0xA9; split the line so the two bytes land in different chunks.
    const full = new TextEncoder().encode('{"t":"café"}\n');
    const splitAt = full.length - 3; // mid-way through the é byte pair
    const out: unknown[] = [];
    for await (const obj of parseNdjson(readerFromBytes([full.slice(0, splitAt), full.slice(splitAt)]))) {
      out.push(obj);
    }
    expect(out).toEqual([{ t: 'café' }]);
  });
});
