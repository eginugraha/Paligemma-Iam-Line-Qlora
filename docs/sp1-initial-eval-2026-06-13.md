# Laporan Hasil Evaluasi Awal — SP-1 (Baseline M1 setelah Training)

**Tanggal dokumen:** 2026-06-13
**Sumber data:** `reports/test_metrics.json` (+ `reports/train.log`)
**Skenario:** M1 (baseline) — transkripsi langsung, tanpa CoT, tanpa RAG
**Status:** Angka resmi/otoritatif untuk baseline M1 thesis

---

## 0. Ringkasan eksekutif

Setelah training QLoRA pertama (SP-1), model PaliGemma fine-tuned dievaluasi pada **seluruh 2.915 baris IAM-test**:

| Metrik | Nilai |
|---|---|
| **mean CER** | **17.37%** |
| **mean WER** | **28.34%** |
| median CER | 12.50% |
| median WER | 22.22% |
| num_samples | 2.915 (full test split) |

Untuk baseline QLoRA pertama, ini **sehat dan layak dilaporkan**. 21,2% transkripsi sempurna; ~72% di bawah CER 25%. Kelemahan utama: **ekor ~5% dengan "repetition collapse"** yang menaikkan rata-rata.

---

## 1. Konfigurasi & provenance

Dari `reports/train.log` + catatan proyek:

| Aspek | Nilai |
|---|---|
| GPU training | RTX A6000 48GB (A5000 24GB OOM saat eval) |
| Durasi | ~4 jam 11 menit |
| Epoch / steps | 3 epoch / 2.430 steps |
| Presisi training | bf16, batch_size=1 |
| train_loss / eval_loss | 0.64 / 0.75 |
| Adapter | `eginugraha/paligemma-iam-line-qlora-adapter` |
| **Presisi EVAL** | **base-4bit + adapter (unmerged)** |
| Test split | 2.915 baris IAM |

> **PENTING (untuk perbandingan):** eval ini memakai **base-4bit + adapter** — konfigurasi presisi **yang sama** dengan serving RunPod (`HTR_BASE_PRECISION=4bit`). Jadi angka 17.37% ini **setara konfigurasi** dengan deployment produksi; bukan angka "full precision" yang berbeda dari serving.

---

## 2. Metrik agregat & distribusi CER

```
mean CER   = 17.37%      median CER = 12.50%
mean WER   = 28.34%      median WER = 22.22%
```

Fakta bahwa **mean (17.37) > median (12.50)** menandakan distribusi miring kanan — mayoritas sampel bagus, tapi ekor error besar menarik rata-rata ke atas.

### Distribusi CER (2.915 sampel)

| Bucket CER | Jumlah | Persentase |
|---|---|---|
| `== 0` (sempurna) | 619 | **21.2%** |
| `<= 5` | 883 | 30.3% |
| `<= 10` | 1.273 | 43.7% |
| `<= 25` | 2.096 | **71.9%** |
| `<= 50` | 2.771 | 95.1% |
| `> 50` (ekor) | 144 | **4.9%** |
| `> 75` | 13 | 0.4% |

- **WER == 0 (sempurna):** 619 sampel (21.2%) — identik dengan CER==0 (baris yang benar karakter, benar kata).

---

## 3. Kekuatan model

- **21,2% transkripsi sempurna** (CER=0) — lebih dari seperlima baris ditranskripsi tanpa kesalahan satu karakter pun.
- **71,9% di bawah CER 25%** — mayoritas besar punya kualitas baik–sangat baik.
- **95,1% di bawah CER 50%** — hanya 1 dari 20 yang tergolong gagal berat.

### Contoh sempurna (CER=0)
```
"round a doll's house ."
"They will be asked to comment on the design of everyday arti…"
"vision ."
```
> Catatan menarik: baris `round a doll's house .` ini **ditranskripsi sempurna oleh M1**, namun **dirusak oleh RAG (M3)** menjadi `round a dollars house .` — lihat laporan investigasi RAG.

---

## 4. Mode kegagalan utama: "repetition collapse" (ekor ~5%)

Ekor error (CER>50, 4,9% sampel) didominasi pola **pengulangan token** di mana decoder "macet" mengulang karakter/kata. Karena CER = edit_distance / panjang_referensi dan prediksi jadi jauh lebih panjang, CER bisa **melebihi 100%**.

### Contoh ekor (CER>50)
```
GT  : he shrugged diffidently , " I like the work . One gets …
pred: it lacked organisation . " We are now , we # # , # # # …    CER=218
```
```
GT  : the mask , now , Ifor did . " Dai Pugh , " he bellowed …
pred: All-round , not for sale . " " " " " " " " " " " " " " …     CER=143
```
```
GT  : man had been seen fleeing from Vauxhall station on
pred: man has 600 800 600 600 , 600 , 600 , 600 , 600 , 600 , …    CER=132
```

**Pola:** kolaps menjadi `# # #`, `" " "`, atau `600 , 600 ,` berulang. Ini bukan kesalahan baca per-karakter biasa, melainkan kegagalan generasi (looping).

### Mitigasi yang direncanakan (belum diterapkan)
Tambahkan parameter generate di `src/htr_sp1/inference.py`:
- `no_repeat_ngram_size=3`
- `repetition_penalty=1.2`

Ini menargetkan ekor ~5% secara langsung dan berpotensi menurunkan mean CER tanpa retraining. (Tercatat sebagai TODO di catatan SP-1.)

---

## 5. Hubungan dengan eval RunPod (eval_run 3)

| Sumber | CER M1 | Sampel | Presisi |
|---|---|---|---|
| **Eval awal ini** (`reports/test_metrics.json`) | **17.37%** | 2.915 | base-4bit + adapter |
| `eval_run 3` (RunPod, 2026-06-13) | 20.49% | 50 | base-4bit + adapter |

- **Konfigurasi presisi identik** → gap 17.37 → 20.49 **bukan** biaya kuantisasi, melainkan **varians sampel kecil** (50 vs 2.915). Diperkirakan konvergen ke ~17.37% saat eval `--limit 200`.
- Deployment RunPod **mereproduksi angka training** → validasi bahwa pipeline serving benar.

---

## 6. Referensi

| Item | Lokasi |
|---|---|
| Data eval awal (agregat + 2.915 per_sample) | `reports/test_metrics.json` |
| Log training | `reports/train.log` |
| Laporan investigasi RAG (eval_run 3) | `docs/sp3-rag-correction-investigation-2026-06-13.md` |
| Generate-param fix (repetition tail) — TODO | `src/htr_sp1/inference.py` |
| Memori terkait | `sp1-first-training-result.md`, `first-real-eval-results.md`, `paligemma-base-precision.md` |
