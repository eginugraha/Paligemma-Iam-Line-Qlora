# Summary — Hasil Perbandingan Akhir & Kesimpulan

> Dokumen rangkuman: menyatukan keempat skenario (M1–M4), menyajikan tabel
> perbandingan, dan menarik kesimpulan utama untuk tesis. Untuk detail tiap
> komponen, lihat dokumen masing-masing yang ditaut di bawah.

## Table of Contents

1. [Description](#1-description)
2. [Logika / Algoritma (gambaran sistem)](#2-logika--algoritma-gambaran-sistem)
3. [Glosarium](#3-glosarium)
4. [Report — Perbandingan & Kesimpulan](#4-report--perbandingan--kesimpulan)
   - 4.1 [Tabel perbandingan 4 skenario](#41-tabel-perbandingan-4-skenario)
   - 4.2 [Temuan utama](#42-temuan-utama)
   - 4.3 [Kesimpulan](#43-kesimpulan)
   - 4.4 [Keterbatasan & arah lanjutan](#44-keterbatasan--arah-lanjutan)

---

## 1. Description

Tesis ini membandingkan **empat skenario** Handwritten Text Recognition pada model
**PaliGemma-3B** yang sama, untuk mengisolasi kontribusi tiap teknik:

| Skenario | Nama | Teknik | Dokumen |
|----------|------|--------|---------|
| **M1** | Baseline | QLoRA fine-tuning | [QLoRA](qlora.md) |
| **M2** | Reasoned | QLoRA + CoT (prompt-only) | [QLoRA + CoT](qlora-cot.md) |
| **M3** | Corrected | QLoRA + RAG (koreksi leksikal) | [QLoRA + RAG](qlora-rag.md) |
| **M4** | Optimal | Hybrid: QLoRA + CoT + RAG | [Hybrid](hybrid.md) |

Desainnya **faktorial 2×2**: sumbu *prompting* (Langsung vs CoT) × sumbu
*post-processing* (Tanpa RAG vs Dengan RAG).

---

## 2. Logika / Algoritma (gambaran sistem)

```
                          gambar baris tulisan tangan
                                      │
                       PaliGemma-3B (QLoRA, base-4bit + adapter)
                          │                           │
          prompt langsung │                           │ prompt CoT
                          ▼                           ▼
                     M1 (baseline)               M2 (CoT, parse 'Final:')
                          │                           │
              korektor RAG│ (pgvector + Levenshtein)  │ korektor RAG
                          ▼                           ▼
                     M3 = RAG(M1)               M4 = RAG(M2)
```

- Satu model, dua prompt (M1 langsung, M2 CoT).
- Satu korektor RAG dipakai dua kali (M3 atas M1, M4 atas M2).
- Orkestrasi & isolasi kegagalan: `src/htr_sp2/orchestrator.py`.

---

## 3. Glosarium

> Istilah teknis lengkap ada di tiap dokumen komponen. Ringkasan paling penting:

| Istilah | Penjelasan singkat |
|---------|--------------------|
| **CER / WER** | Character / Word Error Rate (0% = sempurna), diukur via `jiwer`. |
| **QLoRA** | Fine-tuning hemat memori: base 4-bit beku + adapter LoRA kecil. → [detail](qlora.md#3-glosarium) |
| **CoT** | Chain-of-Thought, prompting penalaran sebelum jawaban. → [detail](qlora-cot.md#3-glosarium) |
| **RAG** | Koreksi leksikal berbasis retrieval kosakata di pgvector. → [detail](qlora-rag.md#3-glosarium) |
| **Desain faktorial 2×2** | Menyilangkan dua faktor → empat sel (M1–M4) untuk isolasi efek. |
| **safe-but-not-helpful** | Konfigurasi yang tidak merusak tetapi juga tidak menambah nilai. |
| **base-4bit + adapter** | Konfigurasi presisi serving produksi; angka eval setara konfigurasi ini. |

---

## 4. Report — Perbandingan & Kesimpulan

### 4.1 Tabel perbandingan 4 skenario

| Skenario | Teknik | Efek pada akurasi | Status |
|----------|--------|-------------------|--------|
| **M1** | QLoRA | **Baseline** — mean CER **17,37%**, WER **28,34%** (test, 2.915 sampel, base-4bit) | Angka resmi |
| **M2** | + CoT | **≈ baseline** — prompt penalaran tidak memberi peningkatan berarti | Net-nol |
| **M3** | + RAG | **≈ baseline** — threshold terbaik **T=0,0** (nyaris tanpa koreksi); T besar → memburuk | Safe-but-not-helpful |
| **M4** | Hybrid | **≈ baseline** — mewarisi keterbatasan M2 & M3 | Net-nol |

> **Catatan metodologis (jujur):** angka 17,37% adalah baseline M1 resmi pada split
> **test**. Kalibrasi RAG dilakukan pada **set berbeda** (100 prediksi M1 split
> validation), sehingga angka absolutnya (mis. 14,35% pada set itu) **tidak**
> sebanding apple-to-apple dengan 17,37%. Yang konsisten lintas set adalah **arah
> efeknya**: CoT, RAG, dan Hybrid sama-sama **tidak** mengungguli baseline secara
> net. Untuk klaim akhir tesis, jalankan keempat skenario pada **satu set test yang
> sama** lalu isi ulang kolom CER/WER tabel ini.

### 4.2 Temuan utama

1. **Kontribusi akurasi dominan datang dari QLoRA fine-tuning itu sendiri.**
   Mengkhususkan PaliGemma untuk transkripsi IAM menghasilkan baseline yang sehat
   (21,2% baris sempurna; 71,9% di bawah CER 25%).
2. **CoT prompt-only tidak otomatis membantu** model yang di-fine-tune khusus
   transkripsi — karena penalaran adalah tugas *out-of-distribution* baginya.
3. **RAG leksikal "aman tapi tidak membantu":** dengan gate Option B + kalibrasi,
   ia berhenti merusak (over-correction teratasi) tetapi optimum justru di T=0,0 —
   yakni tidak mengoreksi. Akarnya: **store berkosakata tertutup (train-only)** +
   **keputusan berbasis edit-distance murni tanpa sinyal konteks**.
4. **Hybrid bukan jumlah dari perbaikan:** menumpuk dua lapisan net-nol tetap
   net-nol. Perbaikan post-hoc tidak selalu komposabel.

### 4.3 Kesimpulan

Pada setup ini, **fine-tuning (QLoRA) adalah pengungkit akurasi utama**, sedangkan
**CoT dan RAG tidak memberi peningkatan net** pada keluaran HTR PaliGemma-3B yang
terkuantisasi. Ini **bukan hasil negatif yang gagal**, melainkan temuan ilmiah yang
sah dan dapat dipertahankan: ia memetakan **kapan** teknik post-hoc populer
(reasoning prompt, koreksi leksikal) **tidak** menambah nilai, beserta alasan
mekanistiknya.

### 4.4 Keterbatasan & arah lanjutan

- **CoT:** model tidak dilatih menalar; CoT terstruktur (deteksi ambiguitas +
  rekomendasi kandidat) kemungkinan butuh **training khusus** dan datanya mahal
  dibuat. Keandalan pada model **4-bit** terbatas (kalibrasi keyakinan menurun).
- **RAG:** untuk benar-benar membantu, perlu **melebarkan store kandidat** ke kamus
  umum **dan** menambah **sinyal plausibilitas** (prior frekuensi / margin / konteks)
  di luar edit-distance murni. Tanpa itu, threshold hanyalah katup antara "diam" dan
  "merusak".
- **Metodologi:** samakan set evaluasi keempat skenario agar tabel §4.1 dapat diisi
  dengan angka yang sebanding.

**File terkait:**
- [`docs/sp1-initial-eval-2026-06-13.md`](../sp1-initial-eval-2026-06-13.md) — baseline M1
- [`docs/sp3-rag-correction-fix-2026-06-15.md`](../sp3-rag-correction-fix-2026-06-15.md) — kalibrasi RAG
- `reports/test_metrics.json`, `reports/tune_sp3_english.json`

**Lihat juga:** [QLoRA](qlora.md) · [QLoRA + CoT](qlora-cot.md) · [QLoRA + RAG](qlora-rag.md) · [Hybrid](hybrid.md)
