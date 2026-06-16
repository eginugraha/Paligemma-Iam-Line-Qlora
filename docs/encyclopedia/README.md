# Ensiklopedia HTR — PaliGemma · QLoRA · CoT · RAG

Kumpulan dokumen referensi (encyclopedia) untuk tesis perbandingan 4 skenario
Handwritten Text Recognition (HTR) berbasis **PaliGemma-3B**. Setiap dokumen
berdiri sendiri: bisa dibaca terpisah untuk menjelaskan satu komponen, dan
saling tertaut untuk gambaran utuh.

## Peta dokumen

| # | Dokumen | Isi singkat |
|---|---------|-------------|
| 0 | [Summary — Perbandingan Akhir & Kesimpulan](summary.md) | Tabel perbandingan M1–M4, temuan utama, kesimpulan tesis |
| 1 | [QLoRA](qlora.md) | Fine-tuning PaliGemma 4-bit (skenario **M1 / baseline**) |
| 2 | [QLoRA + CoT](qlora-cot.md) | Chain-of-Thought prompt-only (skenario **M2**) |
| 3 | [QLoRA + RAG](qlora-rag.md) | Koreksi leksikal berbasis pgvector (skenario **M3**) |
| 4 | [Hybrid (QLoRA + CoT + RAG)](hybrid.md) | CoT lalu dikoreksi RAG (skenario **M4**) |

## Pemetaan skenario ke kode

```
M1 (QLoRA baseline)  ── htr_sp1 (training) + htr_sp2 engine (serving)
M2 (QLoRA + CoT)     ── htr_sp2/cot.py (prompt + parser)
M3 (QLoRA + RAG)     ── htr_sp3/* (corrector) ← koreksi teks M1
M4 (Hybrid)          ── htr_sp3/* ← koreksi teks M2

                 ┌─ M3 = RAG(M1)   [tag: "Corrected"]
   M1 ───────────┤
   M2 (CoT) ─────┴─ M4 = RAG(M2)   [tag: "Optimal"]
```

Orkestrasi keempatnya: `src/htr_sp2/orchestrator.py`.

## Konvensi

- **CER** = Character Error Rate, **WER** = Word Error Rate (lewat `jiwer`, skala 0–100%).
- Angka resmi baseline diukur pada konfigurasi **base-4bit + adapter** — sama persis
  dengan serving produksi (`HTR_BASE_PRECISION=4bit`), bukan angka full-precision.
- Setiap dokumen menautkan **Report** ke file nyata di `docs/` dan `reports/`.
