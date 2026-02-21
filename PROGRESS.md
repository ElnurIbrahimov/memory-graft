# Memory Graft — Progress Log

**Elnur Ibrahimov | February 2026**

---

## What Is Memory Graft?

Surgically splice a trainable memory module into a frozen language model. The base model's weights never change — only the memory block trains. This gives any LLM persistent, updatable memory without fine-tuning.

**Not RAG.** The memory lives inside the forward pass as learned cross-attention at specific transformer layers. No retrieval pipeline, no context window stuffing.

### Architecture

```
Frozen LLM (e.g. Qwen2.5-7B, 7.62B params — NEVER changes)
├── Layers 0-9:    unchanged
├── Layer 10:      [Self-Attn] → [MLP] → [★ MEMORY BLOCK A ★]
├── Layers 11-13:  unchanged
├── Layer 14:      [Self-Attn] → [MLP] → [★ MEMORY BLOCK B ★]
├── Layers 15-17:  unchanged
├── Layer 18:      [Self-Attn] → [MLP] → [★ MEMORY BLOCK C ★]
├── Layers 19-27:  unchanged
└── LM Head

Each Memory Block (~11M params, TRAINABLE):
├── READ pathway:
│   ├── query_proj: Linear(d_model → memory_dim)
│   ├── cross_attn: MultiheadAttention(memory_dim, 8 heads)
│   ├── output_proj: Linear(memory_dim → d_model)
│   ├── gate: scalar (ReZero, init=0.01)
│   └── layer_norm: LayerNorm(d_model)
├── WRITE pathway:
│   ├── key_proj: Linear(d_model → memory_dim)
│   ├── value_proj: Linear(d_model → memory_dim)
│   └── importance_gate: MLP(d_model → d_model/4 → 1)
└── RECONSTRUCTION (auxiliary):
    └── recon_proj: Linear(memory_dim → d_model)
```

### How It Works

**Writing:** Run a fact through the frozen model. Capture hidden states at each target layer. Each memory block's write pathway projects the last token to (key, value) pairs stored in a memory bank.

**Reading:** During inference, the hook at each target layer projects the current hidden state to a query, cross-attends to the memory bank, and adds the result via a gated residual connection.

**Training:** Joint end-to-end training of read + write pathways. Loss = LM loss (next-token prediction) + reconstruction loss (can we recover the original hidden state from the stored value?) + contrastive loss (push different facts apart in memory space).

---

## Experiment History

### Phase 1: Read-Only Training (TinyLlama 1.1B)

**Goal:** Prove the read pathway works — can the model use injected memory?

- Model: TinyLlama-1.1B-Chat, layer 11, memory_dim=512
- Froze write pathway, pre-encoded facts with random projections
- Trained only: query_proj, cross_attn, output_proj, gate, layer_norm (~4.7M params)
- **Result:** Model learned to read from memory. Gate opened from 0.01 → 0.13.
- **Problem:** Random write projections = garbage in memory. Read pathway learned but had nothing useful to read.

### Phase 2: Write Pathway Experiments (Qwen 7B)

Switched to Qwen2.5-7B-Instruct for more capacity. Layer 14, memory_dim=512 → 1024.

**v1-v3: Staged training (Phase 1 then Phase 2)**
- Train read pathway first, then unfreeze write pathway
- Chicken-and-egg problem: read pathway overfits to bad write encodings, write pathway can't improve because read pathway is stuck
- All showed representation collapse — every fact encoded to same point

**v4: Mean-pool encoding**
- `encode_to_memory` averaged all token embeddings
- "The user's name is Alice" and "The user's name is Bob" → nearly identical keys
- Discriminative tokens diluted by shared prefix

**v5: Last-token encoding**
- Switched to using final token only (captures full autoregressive context)
- Better separation between facts, but still collapsed during training

**v6: Contrastive loss**
- Added loss pushing different fact encodings apart: `sim(key_i, key_j)` for i≠j
- Helped prevent collapse but write pathway still undertrained

**v7: Reconstruction loss**
- Added auxiliary decoder: `recon_proj(value) ≈ original_hidden_state`
- Direct gradient signal to write pathway from step 1
- Bootstraps write learning before read pathway is useful

### Joint v1: Everything Together

**The breakthrough.** Combined all fixes: last-token encoding + contrastive loss + reconstruction loss + joint training (no staged phases).

- Model: Qwen2.5-7B-Instruct, layer 14
- memory_dim=1024, gate_max=0.25, eval_gate=0.2
- 50 epochs, 500 train / 50 test examples
- **Result: +10% memory delta** — first time the model answers better WITH memory than without
- Gate learned to ~0.20, reconstruction loss converged
- Still some collapse on similar templates but much better fact discrimination

---

## Current Status (Joint v1 → v2)

### What Works
- Surgical insertion at arbitrary layers
- Gated cross-attention read pathway
- Joint read+write training from scratch
- Contrastive + reconstruction auxiliary losses
- Gate scaling (ReZero init, clamped max)

### Joint v2 Changes (this commit)
- **Multi-layer injection:** Memory blocks at layers 10, 14, 18 (configurable)
  - Each layer has its own MemoryBlock with independent read/write pathways
  - Hidden states captured at each target layer independently
  - All blocks trained jointly with shared optimizer
- **Updated hyperparameters:**
  - epochs: 50 → 100 (more training for more params)
  - gate_max: 0.25 → 0.4 (allow stronger memory influence)
  - eval_gate: 0.2 → 0.35 (match higher gate ceiling)
  - memory_dim: 1024 → 2048 (less compression, keep 1024 as fallback)
- **CLI:** `--layers 10 14 18` for multi-layer, `--layer 14` still works

### Next Steps
1. Run Joint v2 on RunPod: `python -m mvp.train_joint --device cuda --layers 10 14 18`
2. Compare single-layer vs multi-layer memory delta
3. If multi-layer helps: try different layer combinations (early/mid/late)
4. Scale to harder tasks (multi-hop reasoning, longer facts)

---

## Related Work

- **MemoryLLM** (Wang et al., 2024) — Self-updatable memory pool inside transformer. Similar spirit but different mechanism: they modify attention weights directly. We use separate cross-attention + gated residual.
- **Memory Layers at Scale** (Meta, 2024) — Product-key memory layers as drop-in replacements for FFN layers. Much larger memory (up to 128B params equivalent) but serves a different purpose (knowledge storage vs. personalization).
- **Larimar** (IBM, 2024) — External episodic memory for LLMs using Kanerva machines. Closer to our approach but with a different memory architecture.

### How Memory Graft Differs
- **Frozen base model** — zero fine-tuning, works with any pretrained LLM
- **Surgical forward hook** — no architectural changes to the base model
- **Multi-layer injection** — memory influence at multiple representation depths
- **Tiny parameter count** — ~11M per layer vs billions for the base model

---

## Hardware & Compute

- RunPod H100 PCIe 80GB
- ~30s per epoch for single-layer joint training (500 examples, batch=4)
- Total compute through Joint v1: ~$15-20 of H100 time

---

*Memory Graft — Elnur Ibrahimov & Claude Opus 4.6 — February 2026*
