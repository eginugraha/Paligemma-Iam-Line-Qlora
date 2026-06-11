import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import ScenarioColumn from './ScenarioColumn.svelte';
import type { ResultEvent } from '$lib/types';

const RESULT: ResultEvent = {
  event: 'result',
  model: 'm3',
  text: 'the quick brown fox',
  cer: 0,
  wer: 0,
  latency_seconds: 1.1,
  log: 'pgvector: fux -> fox',
  status_tag: 'Corrected'
};

describe('ScenarioColumn', () => {
  it('renders the filled state with text, badge, latency, log and metrics', () => {
    render(ScenarioColumn, {
      props: { id: 'm3', title: 'QLoRA + RAG', state: 'filled', result: RESULT, hasGroundTruth: true }
    });
    expect(screen.getByText('the quick brown fox')).toBeInTheDocument();
    expect(screen.getByText('Corrected')).toBeInTheDocument();
    expect(screen.getByText(/1\.1/)).toBeInTheDocument();
    expect(screen.getByText(/pgvector: fux -> fox/)).toBeInTheDocument();
    expect(screen.getAllByText('0.00').length).toBeGreaterThan(0);
  });

  it('shows an em dash for CER/WER when there is no ground truth', () => {
    render(ScenarioColumn, {
      props: { id: 'm3', title: 'QLoRA + RAG', state: 'filled', result: { ...RESULT, cer: null, wer: null }, hasGroundTruth: false }
    });
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);
  });

  it('shows a spinner while pending', () => {
    render(ScenarioColumn, { props: { id: 'm1', title: 'Baseline', state: 'pending' } });
    expect(screen.getByTestId('spinner')).toBeInTheDocument();
  });

  it('shows the message when in the error state', () => {
    render(ScenarioColumn, { props: { id: 'm2', title: 'CoT', state: 'error', errorMessage: 'engine timeout' } });
    expect(screen.getByText(/engine timeout/)).toBeInTheDocument();
  });

  it('shows the RAG-off message when disabled', () => {
    render(ScenarioColumn, { props: { id: 'm4', title: 'Optimal', state: 'disabled' } });
    expect(screen.getByText(/not run/)).toBeInTheDocument();
  });
});
