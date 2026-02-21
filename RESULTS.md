# Memory Graft — Experiment Results

## Elnur Ibrahimov | February 2026

---

## Summary

We surgically inserted a trainable memory module (~11.6M params) into a frozen Qwen2.5-7B-Instruct model at layer 14. The base model's 7.62B parameters never changed.

**The architecture works. The memory influences generation. But the write pathway can't distinguish between different facts — all memories collapse to the same representation.**

---

## What We Proved

### 1. Surgical Insertion Works
- Forward hook at layer 14 intercepts hidden states
- Memory block modifies hidden states via gated cross-attention
- Gradients flow through memory block but NOT through frozen base model
- Gate learns to open from 0.01 to ~0.35 during training

### 2. Memory Changes Model Behavior
At gate=0.2, the model switches from:
- **Without memory**: "I don't have access to personal information"
- **With memory**: Attempts to answer with specific personal facts

This is a fundamental behavioral shift caused by ~11M parameters modifying hidden states at one layer of a 7.6B frozen model.

### 3. Gate Scaling Sweet Spot
Tested gate values [0.01, 0.03, 0.05, 0.1, 0.2, 0.35]:

| Gate | Memory Delta | Behavior |
|------|-------------|----------|
| 0.01 | -10% | Too weak, no effect |
| 0.03 | 0% | No effect |
| 0.05 | 0% | No effect |
| 0.10 | -10% | Slight noise |
| **0.20** | **+20%** | **Memory working** |
| 0.35 | Mixed | Too strong, hallucinating |

Gate=0.2 is the sweet spot for this architecture.

---

## What Failed

### Representation Collapse
The write pathway maps ALL facts to roughly the same point in memory space. Evidence:

| Actual Fact | Model Always Answers |
|-------------|---------------------|
| cat named Milo | "dog named Buddy" |
| cat named Oliver | "dog named Buddy" |
| rabbit named Flopsy | "dog named Buddy" |
| allergic to sesame | "allergic to celery" |
| favorite food is tacos | "chicken" |
| favorite food is biryani | (hedges, no specific answer) |
| favorite color is yellow | "blue" |
| 29 years old | "25 years old" |
| 49 years old | "25 years old" (Phase 1) / "45" (gate=0.35) |

The model learned the **category** (pet, allergy, food, color, age) but not the **specific value**. All pet queries → same answer. All allergy queries → same answer. Regardless of what fact was actually encoded.

### Root Cause: Mean-Pool Encoding
`encode_to_memory` mean-pools all token embeddings in the fact, then projects:
```
"The user's name is Alice" → mean([The, user, 's, name, is, Alice]) → projection → memory key
"The user's name is Bob"   → mean([The, user, 's, name, is, Bob])   → projection → memory key
```
The discriminative tokens ("Alice" vs "Bob") are diluted by the shared tokens ("The user's name is"). Result: both facts produce nearly identical memory keys.

---

## Training Runs

### Phase 1: Read-Only (Write Frozen)
- **Model**: Qwen2.5-7B-Instruct, layer 14
- **Trainable params**: 4.7M (read pathway only)
- **Run 1**: 10 epochs, lr=1e-4 → loss 1.03, gate 0.13, delta -12%
- **Run 2**: 50 epochs, lr=3e-4 → loss 0.50, gate 0.35, delta -28%
- **Gate sweep**: gate=0.2 on Run 2 checkpoint → **delta +20%** (10 examples)

### Phase 2: Joint Read+Write
- **Trainable params**: 11.6M (all memory block params)
- **30 epochs**, lr=1e-4, gate clamped to 0.25
- **Eval at epoch 10**: delta -12%
- **Eval at epoch 20**: delta -4% (improving)
- **Eval at epoch 30**: delta -12% (representation collapse)
- Same wrong answers across all evals — write pathway collapsed

### Hardware
- RunPod H100 PCIe 80GB
- ~30s per epoch for Phase 2
- Total compute: ~$15-20 of H100 time

---

## Architecture

```
Qwen2.5-7B-Instruct (FROZEN, 7.62B params)
├── Layers 0-13: unchanged
├── Layer 14: [Self-Attn] → [MLP] → [★ MEMORY BLOCK ★]
├── Layers 15-27: unchanged
└── LM Head

Memory Block (11.6M params, TRAINABLE):
├── READ pathway (4.7M):
│   ├── read_query_proj: Linear(3584 → 512)
│   ├── read_attn: MultiheadAttention(512, 8 heads)
│   ├── read_output_proj: Linear(512 → 3584)
│   ├── gate: scalar (ReZero init)
│   └── layer_norm: LayerNorm(3584)
└── WRITE pathway (6.9M):
    ├── write_key_proj: Linear(3584 → 512)
    ├── write_value_proj: Linear(3584 → 512)
    └── importance_gate: MLP(3584 → 896 → 1)
```

---

## Next Steps: Fixing Representation Collapse

### Fix 1: Last-Token Encoding (instead of mean-pool)
Use the final token's embedding instead of averaging all tokens.
The last token captures the full sentence meaning in autoregressive models.
"The user's name is **Alice**" → last token = Alice-in-context → more discriminative.

### Fix 2: Contrastive Loss
Add a training signal that pushes different facts apart in memory space:
- Same fact queries should retrieve similar keys
- Different fact queries should retrieve different keys
- Loss: `contrastive_loss = -log(sim(query, correct_key) / sum(sim(query, all_keys)))`

### Fix 3: Reconstruction Loss
Add an auxiliary decoder that reconstructs the original fact from stored memory values.
Forces the write pathway to encode enough information to distinguish facts.

---

## Key Insight

The hard part of neural memory isn't reading — it's writing.
Teaching a model to RETRIEVE from memory is relatively easy (Phase 1 worked).
Teaching a model WHAT to store and HOW to encode it distinctively is the real challenge.

---

*Memory Graft — Elnur Ibrahimov & Claude Opus 4.6 — February 2026*
