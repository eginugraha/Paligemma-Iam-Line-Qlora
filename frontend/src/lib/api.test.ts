import { describe, it, expect, vi, afterEach } from 'vitest';
import { detectStream } from './api';
import type { DetectEvent } from './types';

function bodyFrom(lines: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const l of lines) controller.enqueue(enc.encode(l));
      controller.close();
    }
  });
}

async function collect(file: File, gt?: string): Promise<DetectEvent[]> {
  const out: DetectEvent[] = [];
  for await (const e of detectStream(file, gt, 'http://test')) out.push(e);
  return out;
}

afterEach(() => vi.restoreAllMocks());

const PNG = new File([new Uint8Array([1, 2, 3])], 'line.png', { type: 'image/png' });

describe('detectStream', () => {
  it('POSTs multipart and yields parsed events in order', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      body: bodyFrom([
        '{"event":"meta","filename":"line.png","has_ground_truth":true}\n',
        '{"event":"result","model":"m1","text":"hi","cer":0,"wer":0,"latency_seconds":0.5,"log":"x","status_tag":"Raw Output"}\n',
        '{"event":"done"}\n'
      ])
    }));
    vi.stubGlobal('fetch', fetchMock);

    const events = await collect(PNG, 'hi');

    expect(events.map((e) => e.event)).toEqual(['meta', 'result', 'done']);
    // POSTed to the right URL with multipart FormData carrying the file + ground_truth.
    // Cast calls via unknown so TS accepts the (url, init) tuple shape from the zero-arg mock.
    const calls = fetchMock.mock.calls as unknown as Parameters<typeof fetch>[];
    const [url, init] = calls[0]!;
    expect(url).toBe('http://test/v1/detect');
    expect(init!.method).toBe('POST');
    expect(init!.body).toBeInstanceOf(FormData);
    expect((init!.body as FormData).get('file')).toBeInstanceOf(File);
    expect((init!.body as FormData).get('ground_truth')).toBe('hi');
  });

  it('omits ground_truth when not provided', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, body: bodyFrom(['{"event":"done"}\n']) }));
    vi.stubGlobal('fetch', fetchMock);
    await collect(PNG);
    const calls2 = fetchMock.mock.calls as unknown as Parameters<typeof fetch>[];
    expect((calls2[0]![1]!.body as FormData).get('ground_truth')).toBeNull();
  });

  it('throws on a non-OK response (e.g. 422)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 422, body: null })));
    await expect(collect(PNG)).rejects.toThrow(/422/);
  });
});
