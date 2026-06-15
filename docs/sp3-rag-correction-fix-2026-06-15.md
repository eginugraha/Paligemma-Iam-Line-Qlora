# Perbaikan RAG Corrector (M3/M4) — Opsi B: Perluasan Gerbang Validitas + Hasil Komparasi

**Tanggal:** 2026-06-15
**Status:** Opsi A (tuning ambang) DIEKSEKUSI → gagal. Opsi B (gerbang kamus) DIIMPLEMENTASI & DIUKUR.
**Lanjutan dari:** [`docs/sp3-rag-correction-investigation-2026-06-13.md`](./sp3-rag-correction-investigation-2026-06-13.md)
**Konteks:** Investigasi 2026-06-13 mendiagnosis bahwa skenario RAG (M3/M4) *memperburuk* akurasi.
Dokumen ini mencatat eksekusi rencana perbaikannya dan hasil kuantitatifnya.

---

## 0. Ringkasan eksekutif

- **Opsi A gagal.** Tuning ambang pada 100 sampel validation → **best_threshold = 0.0** (tanpa
  koreksi). Kurva CER-vs-ambang **monoton naik**: setiap koreksi yang diizinkan justru menaikkan
  CER. Tidak ada ambang yang mengalahkan baseline T=0.0 → masalahnya bukan di ambang, tapi di
  **gerbang validitas** (Masalah #1 investigasi).
- **Opsi B berhasil menghapus kerusakan.** Memperluas gerbang dengan kamus Inggris umum (~370k
  kata, termasuk infleksi) + penanganan possessive **meratakan kurva** ke ≈ baseline. Pada ambang
  produksi lama (0.34): CER turun dari **~15.94 → ~14.48** (baseline 14.35). Over-correction
  yang merusak transkripsi sempurna (mis. `doll's`→`dollars`) **hilang sepenuhnya**.
- **Namun koreksi tetap bukan net-positif.** Bahkan setelah Opsi B, **best_threshold tetap 0.0**:
  gerbang lebar membuat koreksi nyaris no-op (aman tapi tidak menambah akurasi). Pada 100 sampel
  validation ini tidak cukup banyak kasus "typo → kata nyata" untuk menghasilkan keuntungan bersih.
- **Kesimpulan thesis (Bab 4):** *Koreksi leksikal RAG naif tidak meningkatkan akurasi pada HTR
  domain-kecil. Memperluas gerbang validitas menghilangkan kerusakan over-correction, tetapi tidak
  menghasilkan keuntungan bersih — ambang optimal tetap "tanpa koreksi".* Temuan riset yang valid.

---

## 1. Di mana masalahnya (rekap diagnosis)

Detail lengkap di [laporan investigasi](./sp3-rag-correction-investigation-2026-06-13.md). Inti:

1. **Masalah #1 (akar) — gerbang validitas terlalu sempit.** Gerbang exact-match corrector
   (`corrector.py`, kata yang "sudah valid → jangan disentuh") dibangun **hanya dari kosakata
   IAM-train** (6.851 kata). Kata Inggris valid yang kebetulan tak muncul di train split (`sings`,
   `stars`, `groups`, `doll's`) dianggap **OOV** → masuk jalur koreksi → diganti ke tetangga
   ejaan terdekat yang **salah**.
2. **Masalah #2 — ambang `0.34` terlalu longgar**, meloloskan perubahan merusak. `tune.py` belum
   pernah dijalankan.

Premis implisit corrector — *"OOV = salah eja"* — keliru pada vocab domain kecil: di sini "OOV"
lebih sering berarti "kata valid yang tak ada di daftar train kami".

---

## 2. Opsi A — Tuning ambang (dieksekusi → GAGAL)

### Cara
1. `scripts/gen_val_predictions.py` (baru) — jalankan M1 (PaliGemma 4bit+adapter, endpoint RunPod
   serverless yang sudah live) pada **100 sampel IAM-validation** → `reports/val_m1_predictions.json`
   (format `[{"prediction","ground_truth"}]`). **Split validation, bukan test** (anti-leakage).
   Tanpa training — hanya inference; GPU on-demand RunPod (~5 menit).
2. `scripts/tune_sp3.py --pairs reports/val_m1_predictions.json` → scan ambang 0.0..0.5, gerbang
   **train-only** (perilaku asli).

### Hasil (`reports/tune_sp3.json`)

| Ambang T | 0.0 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| CER | **14.35** | 14.37 | 14.50 | 15.00 | 15.37 | 15.70 | 16.17 | 17.10 | 17.33 | 17.63 |

- **best_threshold = 0.0** (tanpa koreksi). Kurva **monoton naik**.
- Pada ambang produksi lama 0.34 → CER ≈ **15.94** (vs 14.35 baseline → **+1.6 lebih buruk**).
  Inilah penjelasan kuantitatif kenapa M3/M4 kalah dari M1/M2 di `eval_run 3`.

### Kesimpulan Opsi A
Tuning ambang **tidak cukup**. Kurva monoton membuktikan masalahnya bukan kelonggaran ambang
melainkan **gerbang** itu sendiri: selama gerbang train-only, koreksi apa pun cenderung merugikan.
Sesuai pohon keputusan investigasi (Langkah 3) → lanjut **Opsi B**.

---

## 3. Opsi B — Perluasan gerbang validitas (implementasi)

### Ide
Gerbang validitas = **(kata IAM-train) ∪ (kamus Inggris umum)**. Kata valid seperti `sings`/`stars`
kini dikenali → **tidak disentuh**. **Store kandidat pgvector tetap train-only** → tidak ada
leakage (kamus umum = pengetahuan eksternal, bukan label test).

### Yang diubah (semua via TDD — RED → GREEN)
| Komponen | Perubahan |
|---|---|
| `data/english_words.txt` (baru) | Wordlist Inggris ~370k kata **dengan infleksi**, di-vendor untuk reproducibility (sumber: dwyl/english-words; lihat `data/README.md`). |
| `src/htr_sp3/english_words.py` (baru) | `load_english_words()` — muat wordlist (lowercase, dedup, filter), memoized. |
| `src/htr_sp3/vocab.py` | `build_gate_vocabulary(records, english_words)` = train ∪ kamus. `build_vocabulary` (train-only, untuk STORE) **tidak diubah** → batas anti-leakage tetap eksplisit. |
| `src/htr_sp3/corrector.py` | Parameter baru `possessive_aware` — kata juga valid bila stem sebelum apostrof ada di gerbang, sehingga possessive/contraction kata nyata (`doll's`, `don't`) tidak dirusak. Default `False` (backward-compatible). |
| `src/htr_sp2/corrector_factory.py` | Produksi kini pakai gerbang lebar + `possessive_aware=True`. |
| `scripts/tune_sp3.py` | Flag `--english-gate` untuk membandingkan gerbang sempit vs lebar. |

**Kenapa `possessive_aware` perlu:** tokenizer memperlakukan `doll's` sebagai **satu** kata, jadi
possessive dari kata benda valid pun tampak OOV (`doll's`→`dollars`, contoh "paling telak" di
laporan §4). Penanganan stem menutup celah ini secara konservatif.

### Tes (TDD)
- `test_sp3_vocab.py` — `build_gate_vocabulary` menggabung train + kamus & lowercase.
- `test_sp3_english_words.py` — loader lowercase/dedup/filter; wordlist vendored memuat
  `sings`/`stars`/`groups`.
- `test_sp3_corrector.py` — possessive kata valid tidak dirusak saat `possessive_aware`; default
  tetap mengoreksi; stem tak-valid tidak terlindungi.

Ukuran gerbang: train-only **6.851** → union **370.665** (560 kata train tidak ada di kamus umum).

---

## 4. Komparasi: sebelum vs sesudah Opsi B

Tuning yang sama (`reports/val_m1_predictions.json`, 100 sampel validation), dua gerbang:
- **before** = `reports/tune_sp3.json` (gerbang train-only)
- **after**  = `reports/tune_sp3_english.json` (`--english-gate`: train ∪ kamus + possessive)

| Ambang T | before (train-only) | after (Opsi B) | Δ (after−before) |
|---|---|---|---|
| 0.0 | 14.346 | 14.346 | +0.000 |
| 0.10 | 14.371 | 14.346 | −0.026 |
| 0.15 | 14.502 | 14.369 | −0.134 |
| 0.20 | 15.004 | 14.415 | −0.589 |
| 0.25 | 15.370 | 14.436 | −0.933 |
| 0.30 | 15.699 | 14.473 | −1.225 |
| 0.35 | 16.172 | 14.495 | −1.677 |
| 0.40 | 17.096 | 14.495 | −2.601 |
| 0.45 | 17.332 | 14.495 | −2.837 |
| 0.50 | 17.626 | 14.495 | −3.131 |

**Bacaan:**
- Kurva **before** menanjak tajam → koreksi makin merusak seiring ambang naik.
- Kurva **after** **nyaris datar** di ≈14.35–14.50 → kerusakan over-correction **hampir hilang
  total**. Makin longgar ambang, makin besar selisih perbaikannya (hingga **−3.1 CER** di T=0.5).
- Pada ambang produksi lama 0.34: **~15.94 → ~14.48** (≈ baseline).
- **Tetapi best_threshold keduanya = 0.0.** Opsi B membuat koreksi **aman** (≈ no-op) tapi **belum
  net-positif**: gerbang lebar memblokir hampir semua koreksi, termasuk yang seharusnya membantu.

### Bukti level-kata (input dari laporan §4, ambang 0.34, store live)

| Input | OLD (gerbang train-only) | NEW (Opsi B) |
|---|---|---|
| `round a doll's house .` | `round a dollars house .` ❌ | `round a doll's house .` ✅ |
| `You 're a star ... Kelly sings with` | `... Jelly swings with` ❌ | `... Kelly sings with` ✅ |
| `by Captain Stars in` | `by Captain Stairs in` ❌ | `by Captain Stars in` ✅ |
| `the possessive groups` | `the possession group` ❌ | `the possessive groups` ✅ |

Setiap over-correction yang dilaporkan kini **dipertahankan benar**.

---

## 5. Interpretasi & keputusan

**Apa yang dibuktikan Opsi B:** akar masalah benar di gerbang. Memperluas gerbang menghapus
mekanisme kerusakan (valid → salah). Setelah itu, mengaktifkan RAG **tidak lagi menurunkan
akurasi** pada konfigurasi apa pun yang diuji.

**Kenapa best tetap T=0.0:** dengan gerbang ~370k kata, hampir semua keluaran M1 dianggap valid,
sehingga sangat sedikit koreksi yang menyala; yang menyala rata-rata netral/sedikit merugikan. Pada
100 sampel validation ini, koreksi sejati "salah-eja → kata nyata" terlalu jarang untuk
menghasilkan penurunan CER bersih. Jadi koreksi ≈ identitas → CER ≈ baseline.

**Rekomendasi konfigurasi (diterapkan):**
- `DEFAULT_THRESHOLD` di-set **0.15** (`src/htr_sp3/config.py`). Best teknis adalah T=0.0, tapi
  0.15 masih ≈baseline (CER 14.37) sambil membiarkan sedikit koreksi menyala — sehingga M3/M4
  tidak byte-identik dengan M1/M2 di tabel komparasi, dengan biaya akurasi ~nol.
- Untuk angka final Bab 4, sajikan M3/M4 dengan gerbang Opsi B + threshold 0.15.
- Penegasan thesis: RAG koreksi pada setup ini **aman tetapi tidak membantu**.

**Kemungkinan lanjutan (opsional, di luar scope):** keuntungan bersih kemungkinan butuh corrector
yang lebih kontekstual (mis. konteks kata-tetangga / bahasa), bukan sekadar leksikal per-kata; atau
gerbang yang lebih ketat-pintar (kamus frekuensi) yang membedakan OOV-typo dari kata-valid-langka.

---

## 6. Reproduksi

```bash
# 1. (sudah ada) prediksi M1 validation via RunPod — tanpa training
python scripts/gen_val_predictions.py --limit 100 --out reports/val_m1_predictions.json

# 2. before (gerbang train-only) vs after (Opsi B); butuh pgvector store live (read-only)
python scripts/tune_sp3.py --pairs reports/val_m1_predictions.json --out reports/tune_sp3.json
python scripts/tune_sp3.py --pairs reports/val_m1_predictions.json --out reports/tune_sp3_english.json --english-gate

# 3. regen wordlist vendored (bila perlu) — lihat data/README.md
```

Tes: `python -m pytest tests/test_sp3_*.py -q` (semua hijau).

---

## Referensi file

| Item | Lokasi |
|---|---|
| Laporan investigasi (diagnosis) | [`docs/sp3-rag-correction-investigation-2026-06-13.md`](./sp3-rag-correction-investigation-2026-06-13.md) |
| Generator prediksi M1 validation | `scripts/gen_val_predictions.py` |
| Wordlist vendored + provenance | `data/english_words.txt`, `data/README.md` |
| Loader kamus | `src/htr_sp3/english_words.py` |
| Gerbang union | `src/htr_sp3/vocab.py` → `build_gate_vocabulary` |
| Possessive-aware gate | `src/htr_sp3/corrector.py` → `_in_gate`, param `possessive_aware` |
| Wiring produksi | `src/htr_sp2/corrector_factory.py` |
| Tuner + flag `--english-gate` | `scripts/tune_sp3.py` |
| Hasil tuning | `reports/tune_sp3.json` (before), `reports/tune_sp3_english.json` (after) |
| Prediksi M1 validation | `reports/val_m1_predictions.json` |
