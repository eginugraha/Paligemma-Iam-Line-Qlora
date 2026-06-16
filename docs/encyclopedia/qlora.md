# QLoRA — Fine-tuning PaliGemma (Skenario M1 / Baseline)

> Skenario **M1**: transkripsi langsung gambar → teks, tanpa CoT, tanpa RAG.
> Inilah fondasi semua skenario lain — M2/M3/M4 dibangun di atas model hasil QLoRA ini.

## Table of Contents

1. [Description](#1-description)
2. [Logika / Algoritma](#2-logika--algoritma)
   - 2.1 [Kuantisasi 4-bit (Q dari QLoRA)](#21-kuantisasi-4-bit-q-dari-qlora)
   - 2.2 [Adapter LoRA (LoRA dari QLoRA)](#22-adapter-lora-lora-dari-qlora)
   - 2.3 [Alur training end-to-end](#23-alur-training-end-to-end)
   - 2.4 [Inferensi / serving](#24-inferensi--serving)
3. [Glosarium](#3-glosarium)
4. [Report](#4-report)

---

## 1. Description

**QLoRA (Quantized Low-Rank Adaptation)** adalah teknik *fine-tuning* hemat memori
yang memungkinkan melatih model besar di GPU terbatas. Idenya menggabungkan dua hal:

1. **Quantized** — model dasar (PaliGemma-3B, ~2,92 miliar parameter) dimuat dalam
   presisi **4-bit** sehingga bobotnya muat di VRAM kecil. Bobot ini **dibekukan**
   (tidak ikut dilatih).
2. **LoRA** — alih-alih melatih ulang 2,92 miliar parameter, kita menambahkan
   **adapter** kecil (matriks rank rendah) hanya pada sebagian lapisan. Hanya
   parameter adapter yang dilatih — jumlahnya sangat sedikit.

Hasilnya: kita bisa mengkhususkan PaliGemma untuk **transkripsi tulisan tangan**
(dataset IAM-line) dengan biaya komputasi yang jauh lebih murah daripada full
fine-tuning, tanpa mengorbankan banyak akurasi.

Dalam proyek ini, base = `google/paligemma-3b-pt-448` (menerima gambar 448×448 px),
dataset = `Teklia/IAM-line`, dengan prompt tetap `"transcribe the handwritten text\n"`.

### Pembagian dataset (split)

Proyek **tidak** memakai split manual 80/20. Ia memakai **tiga split resmi bawaan**
`Teklia/IAM-line` (`load_iam_splits()` mengembalikan ketiganya apa adanya):

| Split | Jumlah sampel | Proporsi | Fungsi |
|-------|--------------:|---------:|--------|
| **train** | 6.482 | 62,5% | melatih adapter LoRA (`ds["train"]`) |
| **validation** | 976 | 9,4% | memantau loss/overfit selama training (`ds["validation"]`) |
| **test** | 2.915 | 28,1% | evaluasi akhir — angka resmi M1 (`ds["test"]`) |
| **TOTAL** | **10.373** | 100% | |

Memakai split baku IAM (bukan rasio acak buatan sendiri) membuat hasil **sebanding
dengan literatur HTR** dan menjamin **tidak ada kebocoran** antar-split — kosakata
RAG pun dibangun **hanya** dari `train` (lihat [QLoRA + RAG](qlora-rag.md)). Angka
test 2.915 sampel konsisten dengan laporan `sp1-initial-eval`.

**Sumber kode:** `src/htr_sp1/config.py` (hyperparameter), `src/htr_sp1/model.py`
(konfigurasi kuantisasi + LoRA), `src/htr_sp1/data.py` (pemuatan dataset),
`src/htr_sp1/cli.py` (pemakaian split saat training & evaluasi).

---

## 2. Logika / Algoritma

### 2.1 Kuantisasi 4-bit (Q dari QLoRA)

`build_quant_config` / `load_trainable_model` di `model.py` memuat base dengan
`BitsAndBytesConfig` berikut:

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| `load_in_4bit` | `True` | bobot base disimpan 4-bit (bukan 16/32-bit) |
| `bnb_4bit_quant_type` | `"nf4"` | NormalFloat-4: tipe 4-bit yang dioptimalkan untuk bobot ber-distribusi normal |
| `bnb_4bit_use_double_quant` | `True` | *double quantization* — mengkuantisasi konstanta kuantisasi juga; hemat memori ekstra |
| `bnb_4bit_compute_dtype` | `float16` (T4) / `bfloat16` (Ampere/Ada) | komputasi tetap di 16-bit demi stabilitas; dipilih otomatis via `detect_precision()` |

Setelah dimuat, `prepare_model_for_kbit_training(base)` menyiapkan model
ber-kuantisasi untuk dilatih (mengaktifkan *gradient checkpointing*, dll).

### 2.2 Adapter LoRA (LoRA dari QLoRA)

`build_lora_config` mendefinisikan bentuk adapter:

| Parameter | Nilai | Arti |
|-----------|-------|------|
| `LORA_R` | `8` | rank dekomposisi — makin kecil makin sedikit parameter |
| `LORA_ALPHA` | `16` | faktor skala (umumnya 2× rank) |
| `LORA_DROPOUT` | `0.05` | regularisasi pada adapter |
| `LORA_TARGET_MODULES` | `["q_proj","k_proj","v_proj","o_proj"]` | hanya proyeksi *attention* pada language model |

**Keputusan desain penting (didokumentasikan untuk tesis):** vision tower dan
multimodal projector **tidak** di-LoRA. Alasannya: menjaga adapter tetap kecil dan
menghindari ketergantungan pada nama modul internal yang berbeda antar versi
`transformers` (nama salah → crash saat load). Mengadaptasi projector dicatat
sebagai **future lever**, bukan bagian baseline.

`get_peft_model(base, lora)` membungkus base beku + adapter; sejak titik ini, **hanya
parameter LoRA yang trainable**.

### 2.3 Alur training end-to-end

```
1. load_iam_splits()            # ambil train/validation/test dari Hub (datasets)
2. build_training_example()     # gambar → RGB → processor → (input_ids, pixel_values, labels)
3. load_trainable_model()       # base 4-bit beku + adapter LoRA
4. Trainer (HF) berjalan:
      - per_device_train_batch_size = 1
      - gradient_accumulation_steps = 8   → effective batch = 8
      - learning_rate = 2e-4
      - num_train_epochs = 3
      - max_target_tokens = 64            # baris IAM pendek
      - seed = 42                          # reproducibility
5. simpan adapter → push_to_hub (opsional)
```

**Catatan memori (dari pengalaman nyata):** `per_device_eval_batch_size` di-set **1**.
Default HF = 8; karena vocab PaliGemma ~257k, logits eval ber-shape
`[batch, seq, 257k]` → cross-entropy upcast ke fp32 → alokasi ~8 GiB sekaligus → OOM
di GPU 24 GB. (Inilah yang meledak di eval akhir-epoch pertama.) Training final
dilakukan di **RTX A6000**; A5000 24 GB sempat OOM.

### 2.4 Inferensi / serving

`load_for_inference(base_precision=...)` mendukung tiga mode:

- `"4bit"` — muat ulang base 4-bit NF4 **persis seperti saat training**. Inilah yang
  dipakai di produksi (`HTR_BASE_PRECISION=4bit`), sehingga angka eval setara konfigurasi
  dengan serving.
- `"bf16"` / `"fp32"` — base presisi lebih tinggi (membandingkan dampak kuantisasi).

Saat serving, adapter LoRA digabungkan dengan base; engine menerima `(gambar, prompt,
max_new_tokens)` dan mengembalikan teks. M1 memakai prompt transkripsi SP-1 dan cap
~64 token.

---

## 3. Glosarium

| Istilah | Penjelasan |
|---------|------------|
| **PaliGemma** | Vision-Language Model (VLM) Google ~3B parameter; menerima gambar + prompt teks, menghasilkan teks. Varian `-448` menerima input 448×448 px. |
| **Fine-tuning** | Melatih ulang model pra-latih pada data spesifik tugas agar lebih ahli di tugas itu. |
| **Full fine-tuning** | Melatih **semua** parameter model — mahal memori & komputasi. Lawan dari LoRA. |
| **QLoRA** | Quantized LoRA: base ber-kuantisasi 4-bit (beku) + adapter LoRA kecil yang dilatih. |
| **Kuantisasi (Quantization)** | Menyimpan bobot pada presisi lebih rendah (mis. 4-bit) agar muat di memori kecil; sedikit menurunkan akurasi. |
| **NF4 (NormalFloat-4)** | Tipe data 4-bit yang dioptimalkan untuk bobot dengan distribusi mendekati normal. |
| **Double Quantization** | Mengkuantisasi juga konstanta skala kuantisasi → penghematan memori tambahan. |
| **compute_dtype** | Presisi yang dipakai saat **menghitung** (forward/backward), meski bobot disimpan 4-bit. |
| **LoRA (Low-Rank Adaptation)** | Menambahkan matriks rank-rendah (A·B) ke lapisan tertentu; hanya matriks ini yang dilatih. |
| **Rank (r)** | Dimensi dekomposisi LoRA. Kecil = sedikit parameter, murah; besar = kapasitas lebih. |
| **alpha** | Faktor penskalaan kontribusi adapter LoRA. |
| **Adapter** | Modul kecil tambahan yang menyimpan "perubahan" hasil fine-tuning, dipisah dari bobot base. |
| **q/k/v/o_proj** | Proyeksi Query/Key/Value/Output pada mekanisme *attention* — sasaran adapter di sini. |
| **Vision tower** | Bagian model yang meng-encode gambar menjadi fitur visual. |
| **Multimodal projector** | Lapisan yang memetakan fitur visual ke ruang token bahasa. |
| **Gradient accumulation** | Menumpuk gradien dari beberapa langkah sebelum update → meniru batch besar di memori kecil. |
| **Effective batch size** | `per_device_batch × grad_accumulation` = ukuran batch efektif yang "dirasakan" optimizer. |
| **OOM (Out Of Memory)** | GPU kehabisan VRAM → proses gagal. |
| **CER / WER** | Character / Word Error Rate — metrik kesalahan transkripsi (0% = sempurna). |
| **IAM-line** | Dataset benchmark tulisan tangan English, level baris (satu gambar = satu baris). |
| **Repetition collapse** | Patologi decoder: model "macet" mengulang karakter/kata; menaikkan CER drastis. |

---

## 4. Report

**Hasil resmi baseline M1** (lihat [`docs/sp1-initial-eval-2026-06-13.md`](../sp1-initial-eval-2026-06-13.md)),
diukur pada split **test** (2.915 sampel), konfigurasi **base-4bit + adapter** (setara serving):

| Metrik | Nilai |
|--------|-------|
| **mean CER** | **17,37%** |
| **mean WER** | **28,34%** |
| median CER | 12,50% |
| median WER | 22,22% |

Distribusi CER:

| Bucket CER | Persentase |
|-----------|-----------|
| `== 0` (sempurna) | **21,2%** |
| `<= 25` | **71,9%** |
| `<= 50` | 95,1% |
| `> 50` (ekor "repetition collapse") | **4,9%** |

**Interpretasi:** untuk QLoRA baseline pertama, hasil ini sehat dan layak dilaporkan —
lebih dari seperlima baris sempurna, mayoritas besar di bawah CER 25%. Kelemahan
utama: ekor ~5% akibat *repetition collapse*.

**File terkait:**
- Laporan: [`docs/sp1-initial-eval-2026-06-13.md`](../sp1-initial-eval-2026-06-13.md)
- Alur fungsi SP-1: [`docs/sp1-alur-fungsi.md`](../sp1-alur-fungsi.md)
- Perbandingan model: [`docs/trocr-vs-paligemma.md`](../trocr-vs-paligemma.md)
- Metrik mentah: `reports/test_metrics.json`, log training: `reports/train.log`

**Lihat juga:** [Summary](summary.md) · [QLoRA + CoT](qlora-cot.md) · [QLoRA + RAG](qlora-rag.md)
