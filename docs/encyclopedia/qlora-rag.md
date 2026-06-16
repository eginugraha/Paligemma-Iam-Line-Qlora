# QLoRA + RAG — Koreksi Leksikal berbasis pgvector (Skenario M3)

> Skenario **M3**: keluaran transkripsi M1 dilewatkan ke **korektor RAG** yang
> mencoba memperbaiki salah-eja dengan mencari kata terdekat di kosakata
> (vocabulary) IAM-train yang tersimpan sebagai vektor di PostgreSQL/pgvector.

## Table of Contents

1. [Description](#1-description)
2. [Logika / Algoritma](#2-logika--algoritma)
   - 2.1 [Vektorisasi kata (trigram hashing)](#21-vektorisasi-kata-trigram-hashing)
   - 2.2 [Penyimpanan & retrieval (pgvector)](#22-penyimpanan--retrieval-pgvector)
   - 2.3 [Pipeline koreksi per kata](#23-pipeline-koreksi-per-kata)
   - 2.4 [Gate, threshold, dan integritas tesis](#24-gate-threshold-dan-integritas-tesis)
3. [Glosarium](#3-glosarium)
4. [Report](#4-report)

---

## 1. Description

**RAG (Retrieval-Augmented Generation)** dalam konteks ini bukan generasi teks
bebas, melainkan **koreksi leksikal**: untuk tiap kata hasil transkripsi yang
"asing", sistem **mengambil (retrieve)** kandidat kata terdekat dari sebuah
kosakata acuan, lalu memutuskan apakah mengganti kata itu.

Kosakata acuan dibangun dari **split TRAIN IAM** (anti-kebocoran: jawaban
test/validation tidak boleh masuk). Tiap kata diubah menjadi vektor dan disimpan
di **PostgreSQL + ekstensi pgvector**, sehingga pencarian "kata yang ejaannya
mirip" menjadi pencarian *nearest-neighbor* berbasis cosine.

Tujuannya: memperbaiki kesalahan ejaan kecil dari model (mis. `helo` → `hello`)
**tanpa** merusak kata yang sudah benar atau nama diri.

**Sumber kode:** `src/htr_sp3/corrector.py` (otak koreksi),
`src/htr_sp3/vectorize.py` (vektorisasi), `src/htr_sp3/store.py` (pgvector),
`src/htr_sp3/vocab.py` (pembentukan kosakata & gate),
`src/htr_sp2/corrector_factory.py` (perakitan korektor).

---

## 2. Logika / Algoritma

### 2.1 Vektorisasi kata (trigram hashing)

`vectorize.word_to_vector(word)`:
1. lowercase, lalu beri **padding batas** `#` (agar awalan/akhiran punya trigram
   sendiri, mis. `#me` ≠ `me` di tengah kata),
2. potong menjadi **trigram karakter** (`NGRAM_N = 3`),
3. tiap trigram di-**feature-hash** dengan **md5** (deterministik lintas proses,
   tidak seperti `hash()` Python yang ber-salt) ke salah satu dari
   `VECTOR_DIM = 512` bucket; hitung frekuensinya,
4. **L2-normalisasi** → cosine similarity = dot product.

Intuisi: kata yang berbagi banyak trigram → vektornya mirip → cosine tinggi.

### 2.2 Penyimpanan & retrieval (pgvector)

- **Ingest** (`scripts/ingest_sp3.py` + `htr_sp3/ingest.py`): bangun kosakata dari
  TRAIN (`vocab.build_vocabulary`), vektorkan tiap kata, simpan baris `(word, vec)`
  ke tabel `iam_vocab`, lalu bangun **indeks HNSW** (cosine).
- **Retrieval** (`store.PgVectorStore.nearest`):
  ```sql
  SELECT word, vec <=> %s AS distance FROM iam_vocab
  ORDER BY distance, word LIMIT 5   -- K_NEIGHBORS = 5
  ```
  Operator `<=>` = jarak cosine pgvector. Tie-break `, word` membuat hasil
  **deterministik** dan cocok dengan `InMemoryVectorStore` saat kalibrasi.

> **Fakta penting untuk tesis:** isi store **hanya kata TRAIN** (6.851 kata). Kata
> yang tidak pernah ada di train **tidak mungkin** menjadi kandidat — ini batas
> desain yang menjelaskan kenapa gambar/teks benar-benar baru sulit dikoreksi.

### 2.3 Pipeline koreksi per kata

`RagCorrector.correct(text)` (di `corrector.py`):

```
1. Tokenisasi loss-less (_TOKEN_RE): pecah jadi potongan KATA / NON-KATA bergantian.
   → non-kata (spasi, tanda baca, angka) di-passthrough apa adanya.
2. Untuk tiap KATA (lowercase):
   a. _in_gate(lower)?  → kata valid → BIARKAN (tidak dikoreksi).   [lihat 2.4]
   b. OOV → vectorize → store.nearest(k=5)  → kandidat kasar (cosine).
   c. RERANK kandidat dgn _normalized_levenshtein (edit-distance ÷ panjang terpanjang).
      → ambil jarak terkecil.  ("cosine menyaring, edit-distance memutuskan")
   d. distance <= threshold ?  → ganti dgn kandidat (warisi kapitalisasi via _match_case)
                              ?  → kalau tidak, BIARKAN kata asli (lindungi OOV/nama diri).
3. Join semua potongan → teks terkoreksi. Catat {from, to, distance} per penggantian.
```

`_normalized_levenshtein` menghitung jarak edit (insert/delete/substitusi) dibagi
panjang kata terpanjang, menghasilkan skala 0..1. `_match_case` menyalin pola
kapitalisasi kata asli ke kandidat (ALL-CAPS → upper, Title → capitalize).

### 2.4 Gate, threshold, dan integritas tesis

**Gate** (`_in_gate`) = himpunan kata yang dianggap "sudah valid, jangan disentuh".
Memakai **Option B**: gate = **kata TRAIN ∪ kamus English umum**
(`build_gate_vocabulary`). Pelebaran ini **hanya pada gate**, bukan pada store
kandidat — kamus umum adalah pengetahuan eksternal (bukan label test), sehingga
**tidak ada kebocoran**. Ini memperbaiki *over-correction* kata English valid yang
kebetulan absen dari train (mis. `sings`, `stars`). Mode `possessive_aware` juga
membiarkan posesif/kontraksi kata nyata (`doll's`, `don't`) apa adanya.

**Threshold** (`DEFAULT_THRESHOLD = 0.15`) = jarak edit maksimum agar koreksi
diterima. Inilah katup trilema:
- **besar** → menerima kandidat jauh → *over-correction*,
- **kecil** → nyaris tak pernah memicu → koreksi ≈ 0.

> **Akar masalah (penting):** keputusan koreksi sepenuhnya bergantung pada (a)
> kandidat di store train-only dan (b) **jarak edit murni** — tanpa sinyal "kata
> mana yang lebih masuk akal secara bahasa". Maka "cot" → "cat/cut/cog" jaraknya
> sama dan korektor menebak buta. Threshold tidak menyelesaikan ini; yang kurang
> adalah **sinyal plausibilitas**, bukan ambang.

---

## 3. Glosarium

| Istilah | Penjelasan |
|---------|------------|
| **RAG (Retrieval-Augmented Generation)** | Pola yang memperkaya keluaran model dengan informasi yang **diambil** dari sumber eksternal. Di sini: koreksi berbasis kosakata. |
| **Koreksi leksikal** | Memperbaiki kata berdasarkan ejaan/kamus, bukan berdasarkan makna kalimat. |
| **Vocabulary (kosakata)** | Himpunan kata acuan; di sini dibangun dari split TRAIN IAM. |
| **Embedding / vektor** | Representasi numerik berdimensi tetap dari sebuah objek (kata) agar bisa dibandingkan secara matematis. |
| **n-gram / trigram** | Potongan berurutan n karakter (n=3 → trigram), mis. `hel`, `ell`, `llo`. |
| **Feature hashing** | Memetakan fitur (trigram) ke indeks bucket via fungsi hash — menghemat dimensi tanpa kamus eksplisit. |
| **L2-normalisasi** | Menskalakan vektor agar panjangnya 1, sehingga cosine = dot product. |
| **Cosine similarity / distance** | Ukuran kemiripan arah dua vektor; jarak cosine = 1 − similarity. |
| **pgvector** | Ekstensi PostgreSQL untuk menyimpan & mengindeks vektor + pencarian nearest-neighbor. |
| **HNSW** | Hierarchical Navigable Small World — struktur indeks untuk pencarian nearest-neighbor aproksimatif yang cepat. |
| **Nearest-neighbor (k-NN)** | Mencari k objek terdekat dari sebuah query di ruang vektor. |
| **Levenshtein distance** | Jumlah minimum operasi (insert/delete/substitusi) untuk mengubah satu string ke string lain. |
| **Normalized Levenshtein** | Jarak Levenshtein dibagi panjang terpanjang → skala 0..1. |
| **Rerank** | Mengurutkan ulang kandidat hasil retrieval dengan kriteria kedua (di sini: edit-distance). |
| **Gate** | Himpunan kata "valid" yang dilewati korektor tanpa diubah. |
| **OOV (Out-Of-Vocabulary)** | Kata yang tidak ada di kosakata/gate. |
| **Threshold** | Ambang jarak edit; koreksi diterima hanya bila jarak ≤ threshold. |
| **Over-correction** | Korektor mengubah kata yang sebenarnya sudah benar → memperburuk hasil. |
| **Option B** | Strategi proyek ini: melebarkan **gate** ke train ∪ kamus English (tanpa melebarkan store kandidat). |
| **Kebocoran (leakage)** | Memasukkan informasi jawaban test/validation ke sistem → menggelembungkan metrik secara tidak sah. |

---

## 4. Report

**Temuan utama:** **RAG bersifat "aman tapi tidak membantu"** (*safe-but-not-helpful*).
Setelah perbaikan gate (Option B) dan kalibrasi, kurva CER **mendatar mendekati
baseline** dan threshold terbaik secara teknis adalah **T = 0,0** — artinya konfigurasi
optimal nyaris **tidak melakukan koreksi sama sekali**.

Kurva tuning (pada 100 prediksi M1 split validation, gate Option B —
[`reports/tune_sp3_english.json`](../../reports/tune_sp3_english.json)):

| Threshold | CER |
|-----------|-----|
| **0,00 (terbaik)** | **14,35%** |
| 0,10 | 14,35% |
| 0,15 | 14,37% |
| 0,20 | 14,41% |
| 0,30 | 14,47% |
| 0,50 | 14,49% |

Makin besar threshold, CER **naik** (memburuk) — konfirmasi *over-correction*.
Contoh kerusakan nyata: baris `round a doll's house .` ditranskripsi **sempurna**
oleh M1, tetapi sempat **dirusak** RAG menjadi `round a dollars house .` sebelum
perbaikan `possessive_aware`.

**Penjelasan (untuk pembahasan tesis):** korektor leksikal berkosakata tertutup +
keputusan berbasis edit-distance murni hanya bisa "aman" (T=0 → tidak merusak) atau
"merusak" (T besar). Ia tidak punya sinyal konteks untuk *menambah nilai* secara
konsisten. Ini temuan ilmiah yang sah tentang **batas koreksi leksikal pada keluaran
HTR**.

**File terkait:**
- Perbaikan & kalibrasi RAG: [`docs/sp3-rag-correction-fix-2026-06-15.md`](../sp3-rag-correction-fix-2026-06-15.md)
- Investigasi awal: [`docs/sp3-rag-correction-investigation-2026-06-13.md`](../sp3-rag-correction-investigation-2026-06-13.md)
- Data kurva: `reports/tune_sp3_english.json`, `reports/tune_sp3.json`
- Prediksi sumber: `reports/val_m1_predictions.json`

**Lihat juga:** [QLoRA](qlora.md) · [Hybrid (M4)](hybrid.md) · [Summary](summary.md)
