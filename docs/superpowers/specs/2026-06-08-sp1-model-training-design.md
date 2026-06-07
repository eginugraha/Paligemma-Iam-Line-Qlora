# SP-1 — Model Training & Packaging (Colab) — Design Spec

* **Parent project:** Sistem Analisis Komparatif Pengenalan Tulisan Tangan (Hybrid Vision-LLM + Relational Vector DB) — see `PRD-001.md`
* **Sub-project:** SP-1 of 5 (Model Training & Packaging)
* **Date:** 2026-06-08
* **Status:** Approved design — ready for implementation planning

---

## 0. Context: Where SP-1 Fits

The parent PRD describes an end-to-end system that is too large for a single spec. It was decomposed into 5 independently-buildable sub-projects:

1. **SP-1 — Model Training & Packaging (Colab)** ← *this spec*
2. SP-2 — Inference Backend Core (FastAPI): engine abstraction (local/RunPod), M1 + M2 (CoT prompting), `POST /v1/detect` streaming, CER/WER calculator
3. SP-3 — RAG / Vector DB (pgvector): IAM vocabulary, char n-gram vectorizer, cosine query, Levenshtein baseline
4. SP-4 — Frontend (Svelte): upload (dataset picker + free upload + optional GT), 4-column streaming comparison
5. SP-5 — Batch Evaluation + Dashboard: offline batch script → aggregate table → dashboard

**Build order:** SP-1 → SP-2+SP-3 → SP-4 → SP-5. SP-1 is the highest-risk foundation; everything downstream consumes its model artifact.

### Cross-cutting decisions established during brainstorming (apply to all sub-projects)

| Topic | Decision |
|---|---|
| Model lifecycle | Train in Colab → load locally → fallback to RunPod (or similar) if 8GB local is insufficient. Backend (SP-2) uses a swappable inference-engine abstraction (local vs remote = config switch). |
| CoT (M2) | **Prompt-only at inference** (Option A). The model is fine-tuned for transcription only; CoT is a prompting technique in SP-2. This keeps M1/M2 the same model so CoT is the only isolated variable. |
| RAG (M3/M4) | Character n-gram word vectors in pgvector (cosine `<=>`), with pure Levenshtein reported as a baseline comparison. (SP-3) |
| Ground truth | Dataset mode (auto GT from IAM labels) + free-upload mode (optional manually-typed GT; hide CER/WER if absent). (SP-2/SP-4) |
| Batch eval | Offline Python script over IAM → aggregate results to a Postgres table → dashboard reads the table. (SP-5) |
| Execution/UX | M1→M2 are two sequential VLM inferences; M3/M4 are fast RAG lookups. Results stream per-column; the PRD's hard 5-second limit is dropped in favor of per-column progress. (SP-2/SP-4) |

---

## 1. Purpose

Produce a fine-tuned **PaliGemma-3B-PT-448** model (via **QLoRA** on **Teklia/IAM-line**) that transcribes line-level handwriting, **validated by inference in Colab**, **published to the Hugging Face Hub**, accompanied by a **test-set CER/WER report**. This artifact and its documented inference interface are the foundation consumed by SP-2.

## 2. Scope

### In scope
- Fine-tuning PaliGemma-3B-PT-448 for **transcription only** (no CoT in weights).
- Full official Teklia/IAM-line train/val/test splits, with Drive checkpointing.
- CER/WER evaluation on the test split.
- Exporting the LoRA adapter + merged fp16 weights to a private HF Hub repo.
- A documented, reload-validated inference snippet (the interface SP-2 will call).
- A reproducible notebook with pinned dependencies.

### Out of scope (explicit)
- GGUF / MLX conversion.
- `llama-cpp-python` loading or any local runtime packaging.
- RunPod or other remote deployment.
- CoT prompting and RAG (SP-2 / SP-3).
- Any web application code (SP-2 / SP-4 / SP-5).

## 3. Architecture

A single reproducible Colab notebook organized into focused modules/cells. Each module has one clear purpose and a well-defined output consumed by the next.

1. **Setup & Environment**
   - Install and **pin versions** of: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `jiwer` (and supporting libs).
   - Set a global random seed for reproducibility.
   - Mount Google Drive (for checkpoints) and authenticate to the Hugging Face Hub.

2. **Data Module**
   - Load `Teklia/IAM-line` via `datasets` using the **official train/val/test splits**.
   - Preprocess: image → PaliGemma processor at **448px**; build the transcription **prompt prefix**; target = the line's transcription label.
   - Include an inspection cell to visualize a few image/label pairs and confirm preprocessing.

3. **Model Module**
   - Load PaliGemma-3B-PT-448 base in **4-bit (nf4, bitsandbytes)**.
   - Attach **LoRA (peft)** to attention layers + the multimodal projector.
   - Enable gradient checkpointing.

4. **Training Loop**
   - HF `Trainer` (or equivalent) with small per-device batch size + gradient accumulation sized to fit **T4 16GB**.
   - **Checkpoint periodically to Google Drive**, with resume-on-disconnect logic.
   - Evaluate **val-CER each epoch**; train until val-CER stabilizes.

5. **Evaluation Module**
   - Run inference over the **test split**.
   - Compute **CER & WER** with `jiwer` (Levenshtein-based).
   - Emit the baseline metrics table (this is the M1 quality figure for the thesis).

6. **Packaging / Export**
   - Save the LoRA adapter; merge to fp16 weights.
   - **Push adapter + merged weights to a private HF Hub repo.**
   - Document the exact inference call (processor config + prompt + `generate` args) as the SP-2 interface.

7. **Inference Validation (gate)**
   - Reload the model fresh from the Hub.
   - Run a handful of test images and confirm transcriptions are correct.
   - This passing check is the definition of "validated."

## 4. Data Flow

```
IAM (image + label)
  → PaliGemma processor (image tokens + transcription prompt)
  → model (4-bit base + LoRA)
  → generated text
  → CER/WER vs label (jiwer)
```

## 5. Error Handling & Robustness

- **Colab disconnects:** periodic Drive checkpoints + resume-from-checkpoint.
- **OOM on T4:** batch size 1–2, gradient accumulation, gradient checkpointing, bounded max sequence length, 448px fixed resolution.
- **Reproducibility:** deterministic seed + pinned dependency versions.

## 6. Testing / Validation Strategy

- **Sanity overfit:** overfit a tiny batch first (loss → ~0) to verify the pipeline before a full run.
- **Per-epoch val-CER:** early signal that learning is progressing.
- **Final test CER/WER:** the deliverable metric.
- **Reload inference test:** proves the exported artifact loads and infers correctly (packaging correctness).

## 7. Deliverables

1. **Private HF Hub repo** containing the LoRA adapter + merged fp16 weights.
2. **Documented inference snippet** — the interface SP-2 consumes.
3. **CER/WER report** on the IAM test set (baseline M1 quality).
4. **Reproducible Colab notebook** + pinned `requirements`.

## 8. Risks & Open Items

- **PaliGemma 4-bit fit on T4 16GB:** expected to fit (3B in 4-bit ≈ 2–3GB + activations); validate early with the sanity overfit.
- **Colab free disconnects/runtime caps:** mitigated by Drive checkpointing + resume.
- **448px memory pressure:** mitigated by small batch + gradient accumulation.
- **PaliGemma prompt convention:** confirm the correct task-prefix/prompt format for transcription during the data module step.

---

## 9. Definition of Done

SP-1 is done when:
- A full QLoRA run has completed on the official IAM-line splits with stabilized val-CER.
- Test-set CER/WER is computed and reported.
- The LoRA adapter + merged fp16 weights are pushed to a private HF Hub repo.
- A fresh reload + inference on sample images succeeds (validation gate passes).
- The inference interface for SP-2 is documented.
- The notebook reproduces end-to-end with pinned dependencies.
