# QLoRA + CoT — Chain-of-Thought (Skenario M2)

> Skenario **M2**: model QLoRA **yang sama** dengan M1, tetapi diberi prompt
> *Chain-of-Thought* (CoT) yang meminta model "menalar" sebelum menjawab.
> Ini adalah CoT **prompt-only** — tidak ada training atau model tambahan.

## Table of Contents

1. [Description](#1-description)
2. [Logika / Algoritma](#2-logika--algoritma)
   - 2.1 [Prompt CoT](#21-prompt-cot)
   - 2.2 [Parser `parse_cot`](#22-parser-parse_cot)
   - 2.3 [Diagram alur](#23-diagram-alur)
3. [Glosarium](#3-glosarium)
4. [Report](#4-report)

---

## 1. Description

**Chain-of-Thought (CoT)** adalah teknik *prompting* yang meminta model
menuliskan langkah penalaran sebelum jawaban akhir, dengan harapan penalaran
itu meningkatkan kualitas jawaban. Pada model bahasa besar, CoT terbukti
membantu tugas penalaran.

Di sini, hipotesisnya: dengan meminta PaliGemma **mendeskripsikan bentuk goresan
karakter yang ambigu lebih dulu**, lalu menulis transkripsi final, model
diharapkan lebih jarang salah-baca karakter → CER/WER turun.

Hal kunci yang harus dipahami untuk tesis: M2 memakai **model fine-tuned yang
sama persis** dengan M1. Yang berbeda **hanya prompt** dan cara mem-parsing
keluaran. Tidak ada bobot baru, tidak ada training tambahan.

> **Caveat penting:** model ini di-fine-tune **hanya untuk transkripsi**, bukan
> untuk menalar. Maka kemampuan CoT-nya tidak terjamin — model bisa mengabaikan
> format. Desain parser sengaja menangani kasus itu (lihat 2.2).

**Sumber kode:** `src/htr_sp2/cot.py` (prompt + parser),
`src/htr_sp2/orchestrator.py` (pemanggilan), `src/htr_sp2/config.py` (cap token).

---

## 2. Logika / Algoritma

### 2.1 Prompt CoT

`COT_PROMPT` (di `cot.py`):

```
transcribe the handwritten text. First briefly describe the distinctive stroke
shapes of any ambiguous characters. Then on a new line write only the final
transcription prefixed exactly with 'Final:'
```

Tiga instruksi berurutan:
1. transkripsikan teks,
2. **dulu** deskripsikan singkat goresan karakter ambigu,
3. **lalu** di baris baru tulis transkripsi final dengan awalan persis `Final:`.

Cap token M2 = **256** (`M2_MAX_NEW_TOKENS`), jauh lebih besar dari M1 (~64),
karena ada prefiks penalaran sebelum jawaban.

> **Catatan desain (penting untuk dipahami):** bagian "describe ambiguous
> characters" hanyalah **teks bebas** yang ditampilkan sebagai *log*. Ia **tidak**
> distrukturkan, **tidak** menghasilkan daftar kandidat, dan **tidak** memengaruhi
> koreksi. Jadi "analisis ambiguitas" pada implementasi saat ini bersifat naratif
> saja, bukan data.

### 2.2 Parser `parse_cot`

Keluaran model (`raw`) berisi penalaran + `Final: <jawaban>`. `parse_cot(raw)`
membuat **satu keputusan**: apakah penanda `Final:` ada?

**Jalur sukses (ada `Final:`):**
```python
reasoning, _, final = raw.rpartition(FINAL_MARKER)   # split pada Final: TERAKHIR
return final.strip(), reasoning.strip()
```
- `rpartition` memilih kemunculan **terakhir** `Final:` — jika model menulis
  `Final:` beberapa kali, hanya yang paling kanan diambil sebagai jawaban.
- `final` → jawaban (dihitung CER/WER). `reasoning` → log (tooltip frontend).

**Jalur fallback (tidak ada `Final:`):**
```python
stripped = raw.strip()
return stripped, f"{stripped}\n{NO_MARKER_NOTE}"
```
- Model abai format → **seluruh output dianggap jawaban**, agar CER/WER tetap
  mencerminkan keluaran nyata model (bukan kosong/error).
- Log ditandai `"[no 'Final:' marker found — using full output as the
  transcription]"` → kasus ini bisa **difilter & dilaporkan** di tesis.

### 2.3 Diagram alur

```
COT_PROMPT → engine.run (model SAMA dgn M1) → raw
                                                │
                          ┌─────────────────────┴──────────────────────┐
                      "Final:" ada?                                "Final:" tdk ada
                          │                                              │
            jawaban = teks setelah Final: terakhir       jawaban = seluruh output
            log     = teks sebelumnya                    log     = output + NO_MARKER_NOTE
                          └──────────────────┬───────────────────────────┘
                                             ▼
                       text → CER/WER, dan disimpan sebagai m2_text
                              (menjadi SUMBER untuk M4 / Hybrid)
                       log  → ditampilkan di tooltip frontend
```

Di `orchestrator.py`, hasil `text` M2 disimpan sebagai `m2_text` dan kelak menjadi
input koreksi RAG pada skenario **M4** (lihat [Hybrid](hybrid.md)).

---

## 3. Glosarium

| Istilah | Penjelasan |
|---------|------------|
| **Chain-of-Thought (CoT)** | Prompting yang meminta model menulis langkah penalaran sebelum jawaban akhir. |
| **Prompt-only CoT** | CoT yang dicapai murni lewat teks prompt, tanpa melatih/menambah model. |
| **Prompt** | Instruksi teks yang mengkondisikan keluaran model. |
| **Penanda / marker (`Final:`)** | String khusus yang dipakai parser untuk memisahkan penalaran dari jawaban akhir. |
| **`rpartition`** | Operasi string Python: memisahkan pada kemunculan **terakhir** sebuah substring. |
| **Fallback** | Jalur cadangan ketika kondisi ideal (format `Final:`) tidak terpenuhi. |
| **`max_new_tokens`** | Batas jumlah token baru yang boleh dihasilkan model dalam satu generasi. |
| **Reasoning log** | Bagian penalaran model yang disimpan untuk audit/tampilan, bukan untuk metrik. |
| **Out-of-distribution (OOD)** | Tugas/format yang berada di luar data pelatihan model — keandalannya tidak terjamin. |
| **m2_text** | Variabel di orchestrator yang menyimpan jawaban M2; menjadi input M4. |

---

## 4. Report

**Temuan utama:** dalam evaluasi nyata, **CoT (M2) ≈ baseline (M1)** — prompt
penalaran **tidak** memberi peningkatan berarti pada model ini.

**Penjelasan temuan (untuk pembahasan tesis):** ini konsisten dengan caveat di
§1 — model di-fine-tune khusus untuk **transkripsi**, bukan penalaran. Memintanya
menalar lewat prompt saja (tanpa training reasoning) adalah tugas
*out-of-distribution*, sehingga manfaat CoT yang biasa terlihat pada LLM besar
tidak muncul di sini. Hasil "≈ baseline" ini **bukan kegagalan**, melainkan
temuan yang sah: *CoT prompt-only tidak otomatis membantu model HTR khusus*.

**File terkait:**
- Evaluasi awal (baseline pembanding M1): [`docs/sp1-initial-eval-2026-06-13.md`](../sp1-initial-eval-2026-06-13.md)
- Ringkasan perbandingan lengkap: [Summary](summary.md)

**Lihat juga:** [QLoRA](qlora.md) · [Hybrid (M4)](hybrid.md) · [Summary](summary.md)
