# Hybrid — QLoRA + CoT + RAG (Skenario M4)

> Skenario **M4**: jalur "paling lengkap". Keluaran **CoT (M2)** dikoreksi oleh
> **korektor RAG (M3)**. Inilah kombinasi ketiga teknik sekaligus — diberi tag
> antarmuka **"Optimal"** karena secara desain menumpuk semua perbaikan.

## Table of Contents

1. [Description](#1-description)
2. [Logika / Algoritma](#2-logika--algoritma)
   - 2.1 [Komposisi M4 = RAG ∘ CoT](#21-komposisi-m4--rag--cot)
   - 2.2 [Isolasi kegagalan & dependensi](#22-isolasi-kegagalan--dependensi)
3. [Glosarium](#3-glosarium)
4. [Report](#4-report)

---

## 1. Description

**Hybrid (M4)** adalah komposisi fungsi: ambil jawaban dari skenario CoT (M2),
lalu jalankan melalui korektor RAG (sama dengan M3). Secara konsep:

```
M4 = RAG( CoT( gambar ) )
```

Tujuannya menguji apakah **menggabungkan** dua perbaikan post-hoc (penalaran via
prompt + koreksi leksikal) memberi hasil terbaik di antara keempat skenario.
M4 melengkapi desain **faktorial 2×2** tesis:

|              | Tanpa RAG | Dengan RAG |
|--------------|-----------|------------|
| **Langsung** | M1        | M3         |
| **CoT**      | M2        | **M4**     |

Dengan empat sel ini, tesis bisa mengisolasi efek **prompting** (baris) dan efek
**post-processing** (kolom) secara independen.

**Sumber kode:** `src/htr_sp2/orchestrator.py` (perakitan M3/M4),
ditambah seluruh kode [CoT](qlora-cot.md) dan [RAG](qlora-rag.md).

---

## 2. Logika / Algoritma

### 2.1 Komposisi M4 = RAG ∘ CoT

Di `orchestrator.py`, setelah M1 & M2 selesai, blok RAG berjalan bila korektor
tersedia (`ENABLE_RAG`):

```python
_rag_sources = [
    ("m3", m1_text, M3_STATUS_TAG, "m1"),   # M3 = RAG(M1)
    ("m4", m2_text, M4_STATUS_TAG, "m2"),   # M4 = RAG(M2)  ← Hybrid
]
for model, source_text, status_tag, depends_on in _rag_sources:
    ...
    corrected, corrections = corrector.correct(source_text)
```

Jadi M4 **tidak memanggil model lagi** — ia hanya mengoreksi `m2_text` (jawaban CoT
yang sudah dihasilkan). Latensi M4 = waktu `corrector.correct()` saja (query
pgvector + rerank Levenshtein), **bukan** akumulasi inferensi M2. Logika koreksinya
**identik** dengan M3 — lihat [QLoRA + RAG §2.3](qlora-rag.md#23-pipeline-koreksi-per-kata).

Tag status: M3 = `"Corrected"`, M4 = `"Optimal"` (`config.M3_STATUS_TAG` / `M4_STATUS_TAG`).

### 2.2 Isolasi kegagalan & dependensi

M4 **bergantung** pada M2. Orchestrator menangani ini secara eksplisit:

- Jika M2 gagal (mis. `EngineError`) sehingga `m2_text is None`, M4 **dilewati**
  dengan event error `"depends on m2 which failed"` — bukan crash.
- Kegagalan korektor (mis. DB mati) **diisolasi** ke M4 saja lewat `try/except`;
  skenario lain tetap jalan dan stream tetap hidup.

Setiap penggantian dicatat sebagai log `"RAG: <from>→<to> (<distance>)"` (atau "no
corrections") untuk tooltip frontend / lampiran tesis.

---

## 3. Glosarium

> Istilah dasar QLoRA, CoT, dan RAG ada di masing-masing dokumen
> ([QLoRA](qlora.md#3-glosarium), [CoT](qlora-cot.md#3-glosarium),
> [RAG](qlora-rag.md#3-glosarium)). Di bawah hanya istilah khusus M4.

| Istilah | Penjelasan |
|---------|------------|
| **Hybrid (M4)** | Skenario yang menggabungkan QLoRA + CoT + RAG: koreksi RAG atas keluaran CoT. |
| **Komposisi (∘)** | Menerapkan satu proses ke hasil proses lain: `RAG(CoT(x))`. |
| **Desain faktorial 2×2** | Eksperimen yang menyilangkan dua faktor (prompting × post-processing) → 4 sel (M1–M4). |
| **Isolasi kegagalan** | Membuat error satu komponen tidak menjatuhkan keseluruhan pipeline. |
| **Dependensi skenario** | M4 butuh keluaran M2; bila M2 gagal, M4 dilewati dengan rapi. |
| **Status tag** | Label kolom di antarmuka ("Corrected"/"Optimal") untuk membedakan skenario. |

---

## 4. Report

**Temuan utama:** karena M4 = RAG(CoT), ia **mewarisi keterbatasan kedua
komponennya**:
- dari [CoT](qlora-cot.md#4-report): **CoT ≈ baseline** (prompt penalaran tak membantu model HTR khusus),
- dari [RAG](qlora-rag.md#4-report): **RAG safe-but-not-helpful** (threshold terbaik T=0,0).

Akibatnya M4 secara praktis **≈ baseline** juga — menumpuk dua perbaikan yang
masing-masing net-nol tidak menghasilkan lompatan. Tag "Optimal" mencerminkan
**niat desain** (jalur paling lengkap), bukan klaim hasil terbaik yang terbukti.

**Penjelasan (untuk pembahasan tesis):** ini justru hasil yang informatif — ia
menunjukkan bahwa **menambah lapisan post-hoc tidak otomatis komposabel** menjadi
perbaikan, ketika tiap lapisan terbatas pada setup ini. Kontribusi akurasi dominan
tetap berasal dari **QLoRA fine-tuning** itu sendiri.

**File terkait:**
- Komponen: [QLoRA + CoT](qlora-cot.md), [QLoRA + RAG](qlora-rag.md)
- Kalibrasi RAG: [`docs/sp3-rag-correction-fix-2026-06-15.md`](../sp3-rag-correction-fix-2026-06-15.md)
- Perbandingan menyeluruh: [Summary](summary.md)

**Lihat juga:** [Summary](summary.md) · [QLoRA](qlora.md)
