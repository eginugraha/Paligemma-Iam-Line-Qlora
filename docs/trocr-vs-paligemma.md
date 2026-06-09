# TrOCR vs PaliGemma untuk HTR

Catatan perbandingan model untuk pembelaan pilihan model di thesis. Keduanya bisa membaca
tulisan tangan, tetapi jenisnya berbeda total: **TrOCR = OCR spesialis**, **PaliGemma =
Vision-Language Model (VLM) generalis**.

## Perbedaan mendasar: spesialis vs generalis

| | **TrOCR** (`microsoft/trocr-base-handwritten`) | **PaliGemma** (`google/paligemma-3b-pt-448`) |
|---|---|---|
| Jenis | OCR specialist (encoder–decoder) | Vision-Language Model (VLM) |
| Arsitektur | ViT image encoder + decoder teks | SigLIP vision encoder + **LLM Gemma** |
| Ukuran | ~334 juta parameter | **~2,92 miliar** parameter (~10x lebih besar) |
| Dilatih untuk | **khusus** transkripsi (sudah di-tune di IAM handwriting) | umum (VQA, captioning, OCR, deteksi, dll.) |
| Input | gambar → teks. Selesai. | **prompt teks + gambar** → teks |
| Bisa diberi instruksi? | ❌ tidak | ✅ ya |
| Bisa Chain-of-Thought (CoT) / reasoning? | ❌ tidak | ✅ ya |
| Bisa konsumsi konteks RAG? | ❌ tidak | ✅ ya |

## Analogi

- **TrOCR** = mesin fotokopi-ke-teks yang sangat jago. Satu tugas, dikerjakan dengan baik.
  Tidak bisa diajak "berpikir".
- **PaliGemma** = asisten yang *melihat* gambar **dan** punya otak bahasa (LLM di dalamnya).
  Bisa disuruh: "baca ini, lalu jelaskan langkahmu" (CoT) atau "baca ini, dengan
  mempertimbangkan contoh berikut" (RAG).

## Kenapa ini krusial untuk thesis ini

Thesis membandingkan **4 skenario**: M1 baseline, **M2 +CoT**, **M3 +RAG**, M4 hybrid.
Inti kebaruan (novelty) penelitiannya: *apakah reasoning (CoT) dan retrieval (RAG)
memperbaiki HTR?*

→ M2 dan M3 **mustahil dilakukan dengan TrOCR.** TrOCR tidak bisa mengikuti instruksi, tidak
bisa Chain-of-Thought, dan tidak bisa menerima konteks retrieval. Ia hanya transkripsi murni.

**Jadi PaliGemma bukan sekadar pilihan — ia syarat mutlak desain penelitian.** Tanpa LLM di
dalamnya, tiga dari empat skenario tidak akan ada.

Teman yang memakai TrOCR kemungkinan bertujuan **HTR murni** (mengejar akurasi transkripsi).
Tujuan thesis ini berbeda: **menguji apakah reasoning + RAG menolong.**

## Trade-off yang jujur (sebaiknya dibahas di thesis)

| Aspek | TrOCR | PaliGemma |
|---|---|---|
| Akurasi transkripsi mentah di IAM | **lebih tinggi** (CER ~4–5%, spesialis + sudah di-tune IAM) | run pertama: CER ~17% |
| Kebutuhan sumber daya | ringan | berat (perlu QLoRA agar muat) |
| Fleksibilitas (instruksi / CoT / RAG) | ❌ | ✅ |

Catatan penting: **TrOCR kemungkinan lebih akurat untuk transkripsi polos.** Jangan
dihindari di thesis — justru jadikan kekuatan argumen:

> "Bukan tujuan kami mengalahkan OCR spesialis pada transkripsi mentah. Kami meneliti apakah
> pendekatan berbasis VLM yang dapat melakukan reasoning (CoT) dan retrieval (RAG) dapat
> **menutup gap** tersebut sekaligus membuka kemampuan yang tidak dimiliki OCR konvensional."

Itulah sebabnya baseline M1 (PaliGemma polos) memang diharapkan lebih lemah daripada TrOCR —
yang menarik adalah **seberapa besar M2/M3/M4 memperbaikinya.**

## Saran (opsional, kuat untuk thesis)

Pertimbangkan menambahkan **TrOCR sebagai baseline pembanding eksternal** di Bab Hasil.
Susunan tabel: TrOCR (referensi spesialis) vs M1–M4 (PaliGemma). Ini menempatkan angka hasil
dalam konteks dan menunjukkan pemahaman atas lanskap model HTR. Kebetulan
`trocr-base-handwritten` sudah ter-cache di mesin lokal, jadi mudah dijalankan untuk
memperoleh angka pembanding.
