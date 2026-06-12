# Investigasi RAG Corrector (M3/M4) — Laporan Temuan & Rencana Perbaikan

**Tanggal:** 2026-06-13
**Status:** Investigasi selesai; perbaikan belum dieksekusi (dilanjutkan besok)
**Konteks:** Eval batch pertama dengan engine RunPod nyata (`eval_run 3`, 50 sampel IAM-test, PaliGemma @4bit) menunjukkan skenario RAG (M3/M4) **memperburuk** akurasi, bukan memperbaiki. Dokumen ini mendiagnosis penyebabnya dan menetapkan rencana perbaikan.

---

## 0. Ringkasan eksekutif

- Pada 50 sampel nyata, **urutan akurasi terbalik dari hipotesis thesis**: M1 (baseline) terbaik, M2 (CoT) ≈ M1, **M3/M4 (RAG) lebih buruk**.
- Corrector mengubah **33 dari 50** sampel; dari yang diubah, **memperburuk 20×** vs **memperbaiki 4×**.
- **Akar masalah:** gerbang validitas kata terlalu sempit (hanya kosakata IAM-train) + ambang koreksi terlalu longgar (`0.34`). Akibatnya kata Inggris **yang sudah benar** dianggap OOV lalu "dikoreksi" ke kosakata mirip yang **salah**.
- **Perbaikan:** (A) tuning ambang via `tune_sp3.py` pada validation; (B) perluas gerbang validitas dengan kamus Inggris umum. Direkomendasikan A dulu, lalu B bila perlu.

---

## 1. Cara kerja corrector saat ini (yang "konservatif")

File: `src/htr_sp3/corrector.py` → kelas `RagCorrector.correct()`.

Corrector **sudah dirancang konservatif** — pipeline per kata:

1. **Tokenisasi** teks jadi potongan word / non-word (`_TOKEN_RE`, baris 29) supaya spasi & tanda baca bisa direkonstruksi verbatim. Token kata mengizinkan apostrof internal (`don't`, `doll's`).
2. **Non-word** (spasi/tanda baca/angka) → lewat tanpa diubah (baris 78-80).
3. **Gerbang exact-match** (baris 84-86): kalau kata (lowercase) **sudah ada di `vocab`** → biarkan apa adanya (idempotent pada kata benar). ← *ini "konservatif"-nya*
4. Kalau kata **OOV** (tidak di vocab): vektorisasi → `store.nearest()` ambil `K_NEIGHBORS=5` kandidat cosine-terdekat.
5. **Rerank** kandidat dengan **normalized Levenshtein** (cosine menyaring, edit-distance memutuskan).
6. **Ganti hanya jika** jarak kandidat terbaik `<= threshold` (baris 89); selain itu **pertahankan kata asli** (melindungi proper noun & OOV asli).
7. Kapitalisasi kata asli diwariskan ke kandidat (`_match_case`, baris 48-54).

**Parameter saat ini** (`src/htr_sp3/config.py`):

| Param | Nilai | Catatan |
|---|---|---|
| `NGRAM_N` | 3 | trigram karakter untuk vektor |
| `VECTOR_DIM` | 512 | feature-hash bucket pgvector |
| `K_NEIGHBORS` | 5 | kandidat sebelum rerank |
| `DEFAULT_THRESHOLD` | **0.34** | "≈1 edit per 3 karakter" — **belum di-tune** |

**Sumber `vocab` gate** (`src/htr_sp2/corrector_factory.py` → `build_vocabulary` di `src/htr_sp3/vocab.py`): **hanya kata dari transkripsi IAM-train** (~6,851 kata). Aturan train-only ini benar untuk **anti-leakage** thesis, tapi dipakai juga sebagai gerbang validitas — inilah sumber masalah (lihat §3).

---

## 2. Bukti kuantitatif

### 2a. Per-skenario (`eval_run 3`, 50 sampel, PaliGemma @4bit)

| Skenario | avg CER | avg WER | latency |
|---|---|---|---|
| **M1 baseline** | **20.49** | **34.91** | 1.80s |
| M2 CoT | 20.90 | 36.85 | 1.85s |
| M3 RAG(M1) | 21.87 | 38.13 | 0.36s |
| M4 RAG(M2) | 22.33 | 40.52 | 0.38s |

> Catatan: CER 20.49% mendekati baseline M1 hasil training SP-1 (**CER 17.37%** pada 2.915 sampel — lihat laporan eval awal: [`docs/sp1-initial-eval-2026-06-13.md`](./sp1-initial-eval-2026-06-13.md)). **Kedua eval memakai konfigurasi presisi yang sama (base-4bit + adapter)**, jadi selisih ~3pt **bukan** biaya kuantisasi melainkan **varians sampel kecil** (50 vs 2.915) — diperkirakan konvergen ke ~17.37% saat `--limit 200`. Jadi modelnya sehat — masalahnya murni di lapisan koreksi RAG.

### 2b. Efek koreksi RAG (M1 → M3, per kata/sampel)

| | Jumlah |
|---|---|
| Teks tidak diubah | 17 / 50 |
| **Teks diubah** | **33 / 50** |
| → memperbaiki (CER turun) | 4 |
| → **memperburuk** (CER naik) | **20** |
| → netral (CER sama) | 9 |

Corrector mengubah **66%** sampel dan **memperburuk 5× lebih sering** daripada memperbaiki.

---

## 3. Akar masalah (dua hal yang saling memperparah)

### Masalah #1 — Gerbang validitas terlalu sempit (penyebab utama)
`vocab` gate dibangun **hanya dari IAM-train** (~6,851 kata). Kata Inggris yang **valid tapi kebetulan tak muncul** di train split (`sings`, `stars`, `doll's`, `groups`) dianggap **OOV** → masuk jalur koreksi → diganti ke kosakata mirip yang salah.

Premis implisit corrector — *"OOV = salah eja"* — **keliru untuk vocab domain kecil**. Di sini "OOV" lebih sering berarti "kata valid yang tidak ada di daftar train kami", bukan "salah eja".

### Masalah #2 — Ambang `0.34` terlalu longgar
Normalized Levenshtein contoh kata yang rusak:
- `sings` → `swings`: 1 edit / 6 = **0.167** ≤ 0.34 → diterima
- `stars` → `stairs`: 1 edit / 6 = **0.167** ≤ 0.34 → diterima
- `doll's` → `dollars`: 2 edit / 7 = **0.286** ≤ 0.34 → diterima

Ambang `0.34` meloloskan perubahan yang merusak. Selain itu **`tune.py` tampaknya belum pernah dijalankan** — `DEFAULT_THRESHOLD` masih nilai tebakan awal, bukan hasil optimasi validation.

---

## 4. Contoh kata yang sudah BENAR malah jadi SALAH

Diambil dari `eval_run 3` (kasus M1 benar/lebih baik → M3 lebih buruk):

| GT (benar) | M1 (sebelum RAG) | M3 (setelah RAG) | Efek |
|---|---|---|---|
| `doll's` | `doll's` (CER **0.0**, sempurna) | `dollars` (CER **9.1**) | **merusak transkripsi sempurna** |
| `sings` | `sings` | `swings` | benar → salah |
| `Kelly` | `Kelly` | `Jelly` | benar → salah |
| `stars` | `Stars` | `Stairs` | benar → salah |
| `possessive` | `possessive` | `possession` | benar → salah |
| `groups` | `groups` | `group` | benar → salah |

### Contoh kalimat lengkap (paling telak)
```
GT : round a doll's house .
M1 : round a doll's house .    CER = 0.0   ← SUDAH SEMPURNA
M3 : round a dollars house .   CER = 9.1   ← RAG merusaknya
```
```
GT : ... You 're a star ... Rolly sings with
M1 : ... You 're a star ... Kelly sings with   CER = 16.9
M3 : ... You 're a star ... Jelly swings with  CER = 18.1   (Kelly→Jelly, sings→swings)
```
```
GT : Fay Compton stars in " No Hiding Place " ...
M1 : by Captain Stars in " No Hiding Place " ...   CER = 25.4
M3 : by Captain Stairs in " No Hiding Place " ...  CER = 27.0   (Stars→Stairs)
```

**Pola jelas:** corrector memaksa kata valid ke "tetangga terdekat" di kosakata kecil, mengganti kata benar dengan kata salah yang mirip secara ejaan.

---

## 5. Opsi perbaikan

### Opsi A — Tuning ambang (paling ringan; tool sudah ada)
Jalankan `scripts/tune_sp3.py` pada prediksi **M1 validation** untuk cari ambang yang meminimalkan CER. Grid menyertakan **T=0.0 (tanpa koreksi)**, jadi hasilnya langsung menjawab *"apakah koreksi bisa membantu sama sekali?"*.

- Ambang `0.15` saja sudah memblokir ketiga contoh rusak (`sings`/`stars` 0.167, `doll's` 0.286).
- Metodologis bersih (validation-only, **bukan** test).
- **Risiko:** ambang terlalu rendah juga memblokir koreksi OOV yang sah; `tune.py` mencari titik optimal.

### Opsi B — Perluas gerbang validitas dengan kamus Inggris umum (perbaikan akar)
Tambahkan word-list Inggris umum (mis. NLTK `words` corpus / `/usr/share/dict/words`) ke gerbang validitas (`vocab` gate), sehingga kata valid seperti `sings`/`stars`/`groups` dikenali → **tidak disentuh**. Store kandidat pgvector tetap **train-only** (tidak ada leakage; kamus umum = pengetahuan eksternal, bukan label test).

- Menyerang akar masalah #1 secara langsung.
- Bisa digabung dengan Opsi A (gerbang lebih ketat + ambang lebih konservatif).

### Rekomendasi
**A dulu** (kuantifikasi & kemungkinan besar sudah memperbaiki mayoritas kerusakan) → kalau masih ada kata valid yang dirusak, tambahkan **B**.

### Implikasi thesis (Bab 4)
Apa pun hasilnya, ini **temuan riset yang valid**, bukan bug pipeline:
> *"Koreksi lexical RAG naif menurunkan akurasi karena OOV ≠ salah eja pada vocab domain kecil (over-correction). Dengan tuning ambang pada validation + gerbang leksikon umum, M3/M4 baru dapat mengungguli M1/M2."*

Jika bahkan ambang optimal tidak mengalahkan T=0.0 → kesimpulan: "RAG correction tidak membantu untuk konfigurasi ini" — tetap kontribusi yang layak dibahas.

---

## 6. Rencana eksekusi (untuk besok)

### Langkah 1 — Generate prediksi M1 validation
- Tulis script kecil: jalankan M1 (PaliGemma via RunPod, `HTR_ENGINE=runpod`) pada ~100 sampel **validation** IAM → simpan `val_m1_predictions.json` berformat `[{"prediction": ..., "ground_truth": ...}]`.
- Butuh **1 GPU run RunPod** (~5 menit + sedikit biaya).
- Gunakan split **validation** (`data.load_iam_splits()["validation"]`), **bukan test** (anti-leakage; test disimpan untuk angka final).

### Langkah 2 — Jalankan tuner (Opsi A)
```bash
python scripts/tune_sp3.py --pairs val_m1_predictions.json --out tune_sp3.json
```
- Output: `best_threshold`, `best_cer`, kurva CER-vs-ambang, dan baseline `T=0.0`.

### Langkah 3 — Interpretasi & keputusan
- Jika `best_cer` < CER(T=0.0) → update `DEFAULT_THRESHOLD` di `src/htr_sp3/config.py` ke `best_threshold`. RAG jadi membantu. ✅
- Jika tidak ada ambang yang mengalahkan T=0.0 → lanjut **Opsi B** (perluas gerbang kamus umum), lalu ulangi Langkah 2.

### Langkah 4 — (Opsi B bila perlu) Perluas gerbang validitas
- Modifikasi `build_vocabulary` / `corrector_factory` agar `vocab` gate = (kata IAM-train) ∪ (kamus Inggris umum). Pertahankan store kandidat tetap train-only.
- Pertimbangkan TDD: tambah test yang memastikan kata valid umum (`sings`, `stars`) tidak diubah.

### Langkah 5 — Eval ulang & konfirmasi
- Jalankan ulang eval kecil (`--limit 50 --rag`) dengan ambang/gerbang baru → bandingkan M3/M4 vs M1/M2.
- Jika M3/M4 sudah ≤ M1/M2 → jalankan eval final `--limit 200` untuk angka Bab 4.

---

## Referensi file & baris

| Item | Lokasi |
|---|---|
| Logika corrector + gerbang + ambang | `src/htr_sp3/corrector.py` (gate baris 84-86; threshold baris 89) |
| Parameter (threshold 0.34, K, ngram) | `src/htr_sp3/config.py` |
| Pembentuk vocab gate (train-only) | `src/htr_sp3/vocab.py` → `build_vocabulary` |
| Pembangun corrector produksi | `src/htr_sp2/corrector_factory.py` → `get_corrector` |
| Logika tuning ambang | `src/htr_sp3/tune.py` → `tune_threshold` |
| CLI tuning | `scripts/tune_sp3.py` |
| Data eval | tabel `eval_result` / `eval_run` di Postgres (`eval_run 3`) |
| **Laporan eval awal (baseline M1 17.37%)** | [`docs/sp1-initial-eval-2026-06-13.md`](./sp1-initial-eval-2026-06-13.md) |
| Memori terkait | `first-real-eval-results.md`, `sp3-rag-pgvector.md`, `runpod-serverless-deploy.md` |
