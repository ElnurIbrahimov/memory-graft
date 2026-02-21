"""
Phase 2: Joint Read+Write Training for Memory Graft.

Key difference from Phase 1:
- Write pathway is UNFROZEN — model learns WHAT to store
- Facts are re-encoded each forward pass with gradients on write projections
- Gradients flow: loss -> read pathway -> cross-attention -> stored keys/values -> write pathway
- Gate clamped to prevent runaway (sweet spot: ~0.2 from Phase 1 experiments)

Usage:
    python -m mvp.train_phase2 --phase1_checkpoint checkpoints/qwen7b_v2/best.pt
"""

import argparse
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm

from .surgery import SurgicalModel
from .memory_bank import MemoryBank
from .data import generate_dataset, format_for_training
from .train import pre_encode_facts, evaluate


def encode_fact_with_grad(surgical_model, text, device):
    """
    Encode a fact into memory keys/values WITH gradients on write pathway.

    Returns (key, value) tensors connected to write_key_proj and write_value_proj.
    """
    surgical_model._memory_active = False
    surgical_model._capture_mode = True
    surgical_model._capture_buffer = None

    input_ids = surgical_model.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).input_ids.to(device)

    with torch.no_grad():
        surgical_model.base_model(input_ids)

    surgical_model._capture_mode = False
    surgical_model._memory_active = True

    h = surgical_model._capture_buffer.to(device)
    if h.dim() == 2:
        h = h.unsqueeze(0)

    # Write pathway WITH gradients (no torch.no_grad here!)
    h_float = h.float()
    keys, values = surgical_model.memory_block.encode_to_memory(h_float)
    h_last = h_float[:, -1, :].detach()  # target for reconstruction loss
    return keys.squeeze(0), values.squeeze(0), h_last.squeeze(0)


def contrastive_key_loss(keys):
    """Push different fact encodings apart in memory space."""
    if keys.size(0) < 2:
        return torch.tensor(0.0, device=keys.device, requires_grad=True)
    keys_norm = F.normalize(keys, dim=-1)
    sim = keys_norm @ keys_norm.T
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    return sim[mask].mean()


def train_phase2(
    surgical_model,
    dataset,
    epochs=30,
    lr=1e-4,
    batch_size=4,
    device="cuda",
    checkpoint_dir="checkpoints",
    gate_max=0.25,
    eval_gate=0.2,
    test_data=None,
    contrastive_weight=0.1,
    n_distractors=0,
    fresh_data=True,
    recon_weight=0.1,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Unfreeze ALL memory block params
    for param in surgical_model.memory_block.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in surgical_model.memory_block.parameters())
    print(f"\nPhase 2: Training ALL {total_params:,} memory block params")
    print(f"  Read + Write pathways. Gate clamped to max {gate_max}.")
    print(f"  Distractors: {n_distractors}, Contrastive: {contrastive_weight}")
    print(f"  Reconstruction weight: {recon_weight}")
    print(f"  Fresh data per epoch: {fresh_data}\n")

    optimizer = AdamW(
        surgical_model.memory_block.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    tokenizer = surgical_model.tokenizer
    base_formatted = [format_for_training(ex, tokenizer) for ex in dataset]
    base_dataset = dataset

    surgical_model.memory_block.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_c_loss = 0.0
        epoch_r_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        # Fresh data each epoch — model can't memorize fixed fact-answer pairs
        if fresh_data:
            dataset = generate_dataset(n=len(base_dataset), seed=42 + epoch)
            formatted = [format_for_training(ex, tokenizer) for ex in dataset]
        else:
            dataset = base_dataset
            formatted = base_formatted

        indices = torch.randperm(len(formatted)).tolist()

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            optimizer.zero_grad()

            # Encode all facts and run forward passes, accumulating loss
            batch_keys = []
            accumulated_loss = None
            batch_lm_total = 0.0

            for idx in batch_indices:
                ex = formatted[idx]
                fact_text = dataset[idx]["fact"]

                # Encode target fact with gradients
                key, value, h_orig = encode_fact_with_grad(surgical_model, fact_text, device)
                batch_keys.append(key)

                # Build memory bank with target + distractors
                bank = MemoryBank(
                    memory_dim=surgical_model.memory_block.memory_dim,
                    max_entries=256,
                )
                bank.write(key, value, detach=False)

                # Add distractor facts — forces selective retrieval
                if n_distractors > 0:
                    d_indices = random.sample(
                        [i for i in range(len(dataset)) if i != idx],
                        min(n_distractors, len(dataset) - 1),
                    )
                    for d_idx in d_indices:
                        d_key, d_value, _ = encode_fact_with_grad(
                            surgical_model, dataset[d_idx]["fact"], device
                        )
                        bank.write(d_key, d_value, detach=False)

                surgical_model.set_memory_bank(bank)

                input_ids = ex["input_ids"].unsqueeze(0).to(device)
                labels = ex["labels"].unsqueeze(0).to(device)

                outputs = surgical_model(input_ids=input_ids, labels=labels)
                lm_loss = outputs.loss / len(batch_indices)
                batch_lm_total += outputs.loss.item()

                # Reconstruction loss — direct gradient to write_value_proj
                h_recon = surgical_model.memory_block.recon_proj(value)
                r_loss = recon_weight * F.mse_loss(h_recon, h_orig) / len(batch_indices)
                epoch_r_loss += F.mse_loss(h_recon, h_orig).item()

                step_loss = lm_loss + r_loss
                if accumulated_loss is None:
                    accumulated_loss = step_loss
                else:
                    accumulated_loss = accumulated_loss + step_loss

                with torch.no_grad():
                    preds = outputs.logits[:, :-1, :].argmax(dim=-1)
                    target = labels[:, 1:]
                    mask = target != -100
                    if mask.any():
                        epoch_correct += (preds[mask] == target[mask]).sum().item()
                        epoch_total += mask.sum().item()

            # Contrastive loss: push different fact keys apart
            if contrastive_weight > 0 and len(batch_keys) > 1:
                keys_tensor = torch.stack(batch_keys)
                c_loss = contrastive_key_loss(keys_tensor)
                accumulated_loss = accumulated_loss + contrastive_weight * c_loss
                epoch_c_loss += c_loss.item()

            accumulated_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                surgical_model.memory_block.parameters(), max_norm=1.0
            )
            optimizer.step()

            with torch.no_grad():
                surgical_model.memory_block.gate.clamp_(max=gate_max)

            epoch_loss += batch_lm_total

        avg_loss = epoch_loss / len(formatted)
        accuracy = epoch_correct / max(epoch_total, 1) * 100
        elapsed = time.time() - t0
        gate_val = surgical_model.memory_block.gate.item()

        n_batches = (len(formatted) + batch_size - 1) // batch_size
        avg_c_loss = epoch_c_loss / max(n_batches, 1)
        avg_r_loss = epoch_r_loss / len(formatted)

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"r_loss={avg_r_loss:.4f}  "
            f"c_loss={avg_c_loss:.4f}  "
            f"acc={accuracy:.1f}%  "
            f"gate={gate_val:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # Save checkpoint
        save_path = checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "memory_block_state": surgical_model.memory_block.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
                "gate": gate_val,
                "phase": 2,
            },
            save_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "memory_block_state": surgical_model.memory_block.state_dict(),
                    "loss": avg_loss,
                    "gate": gate_val,
                    "phase": 2,
                },
                checkpoint_dir / "best.pt",
            )
            print(f"  -> New best (loss={best_loss:.4f})")

        # Evaluate every 10 epochs
        if test_data and (epoch + 1) % 10 == 0:
            print(f"\n  --- Eval at epoch {epoch+1} (gate set to {eval_gate}) ---")
            saved_gate = surgical_model.memory_block.gate.data.clone()
            surgical_model.memory_block.gate.data = torch.tensor(eval_gate)
            test_entries = pre_encode_facts(surgical_model, test_data, device)
            evaluate(surgical_model, test_data, test_entries, device=device)
            surgical_model.memory_block.gate.data = saved_gate
            surgical_model.memory_block.train()
            print()

    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Memory Graft Phase 2")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--memory_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--checkpoint_dir", default="checkpoints/qwen7b_phase2_v5")
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--n_distractors", type=int, default=0)
    parser.add_argument("--fresh_data", action="store_true", default=True)
    parser.add_argument("--phase1_checkpoint", default="checkpoints/qwen7b_v2/best.pt")
    parser.add_argument("--gate_max", type=float, default=0.25)
    parser.add_argument("--eval_gate", type=float, default=0.2)
    args = parser.parse_args()

    # Load model
    sm = SurgicalModel.from_pretrained(
        model_name=args.model,
        layer_idx=args.layer,
        memory_dim=args.memory_dim,
        n_heads=args.n_heads,
        device=args.device,
    )

    # Load Phase 1 checkpoint
    print(f"\nLoading Phase 1 checkpoint: {args.phase1_checkpoint}")
    cp = torch.load(args.phase1_checkpoint, map_location=args.device, weights_only=False)
    sm.memory_block.load_state_dict(cp["memory_block_state"], strict=False)
    print(f"  Phase 1: loss={cp['loss']:.4f}, gate={cp['gate']:.4f}")

    # Generate data
    print(f"\nGenerating {args.n_train} train + {args.n_test} test examples...")
    train_data = generate_dataset(n=args.n_train, seed=42)
    test_data = generate_dataset(n=args.n_test, seed=99)

    # Train Phase 2
    print("\n" + "=" * 60)
    print("PHASE 2: Joint Read+Write Training")
    print("=" * 60)

    train_phase2(
        surgical_model=sm,
        dataset=train_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        gate_max=args.gate_max,
        eval_gate=args.eval_gate,
        test_data=test_data,
        contrastive_weight=args.contrastive_weight,
        recon_weight=args.recon_weight,
        n_distractors=args.n_distractors,
        fresh_data=args.fresh_data,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION (gate={args.eval_gate})")
    print("=" * 60)
    sm.memory_block.gate.data = torch.tensor(args.eval_gate)
    test_entries = pre_encode_facts(sm, test_data, args.device)
    evaluate(sm, test_data, test_entries, device=args.device)

    sm.save_memory_block(f"{args.checkpoint_dir}/memory_block_final.pt")
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
