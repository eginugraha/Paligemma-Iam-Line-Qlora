import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import UploadArea from './UploadArea.svelte';

describe('UploadArea', () => {
  it('emits the selected image file via onfile', async () => {
    const onfile = vi.fn();
    render(UploadArea, { props: { onfile, disabled: false } });

    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const file = new File([new Uint8Array([1])], 'line.png', { type: 'image/png' });
    await fireEvent.change(input, { target: { files: [file] } });

    expect(onfile).toHaveBeenCalledOnce();
    expect(onfile.mock.calls[0][0]).toBe(file);
  });

  it('rejects a non-image file with an inline message and does not emit', async () => {
    const onfile = vi.fn();
    render(UploadArea, { props: { onfile, disabled: false } });

    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const bad = new File(['x'], 'notes.txt', { type: 'text/plain' });
    await fireEvent.change(input, { target: { files: [bad] } });

    expect(onfile).not.toHaveBeenCalled();
    expect(screen.getByText(/png, jpg, or jpeg/i)).toBeInTheDocument();
  });
});
