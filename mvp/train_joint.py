"""
Joint End-to-End Training for Memory Graft.

Trains BOTH read and write pathways simultaneously from scratch.
No Phase 1/Phase 2 split — avoids the chicken-and-egg problem.

Key insight: high reconstruction loss bootstraps the write pathway quickly,
giving the read pathway useful signal to learn from early on.

Supports multi-layer memory injection: trains all memory blocks jointly,
each with its own write pathway encoding facts at its respective layer.

Usage:
    python -m mvp.train_joint --device cuda
    python -m mvp.train_joint --device cuda --layers 10 14 18 --memory_dim 2048
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
    Encode a fact into memory keys/values WITH gradients on write pathways.

    Multi-layer: captures hidden states at all target layers, encodes through
    each layer's memory block independently.

    Returns: dict of {layer_idx: (key, value, h_last)} for each target layer.
    """
    surgical_model._memory_active = False
    surgical_model._capture_mode = True
    for idx in surgical_model.layer_indices:
        surgical_model._capture_buffers[idx] = None

    input_ids = surgical_model.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).input_ids.to(device)

    with torch.no_grad():
        surgical_model.base_model(input_ids)

    surgical_model._capture_mode = False
    surgical_model._memory_active = True

    results = {}
    for idx in surgical_model.layer_indices:
        h = surgical_model._capture_buffers[idx].to(device)
        if h.dim() == 2:
            h = h.unsqueeze(0)

        h_float = h.float()
        block = surgical_model.memory_blocks[idx]
        keys, values = block.encode_to_memory(h_float)
        h_last = h_float[:, -1, :].detach()
        results[idx] = (keys.squeeze(0), values.squeeze(0), h_last.squeeze(0))

    return results


def contrastive_key_loss(keys):
    """Push different fact encodings apart in memory space."""
    if keys.size(0) < 2:
        return torch.tensor(0.0, device=keys.device, requires_grad=True)
    keys_norm = F.normalize(keys, dim=-1)
    sim = keys_norm @ keys_norm.T
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    return sim[mask].mean()


def train_joint(
    surgical_model,
    dataset,
    epochs=100,
    lr=1e-4,
    batch_size=4,
    device="cuda",
    checkpoint_dir="checkpoints",
    gate_max=0.4,
    eval_gate=0.35,
    test_data=None,
    contrastive_weight=0.5,
    recon_weight=0.5,
    fresh_data=True,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ALL params trainable across all memory blocks
    all_params = []
    for idx, block in surgical_model.all_memory_blocks():
        for param in block.parameters():
            param.requires_grad = True
        all_params.extend(block.parameters())

    total_params = sum(p.numel() for p in all_params)
    n_blocks = len(surgical_model.layer_indices)
    print(f"\nJoint training: {total_params:,} params across {n_blocks} layer(s), lr={lr}")
    print(f"  Layers: {surgical_model.layer_indices}")
    print(f"  Gate clamped to max {gate_max}")
    print(f"  Contrastive weight: {contrastive_weight}")
    print(f"  Reconstruction weight: {recon_weight} (HIGH — bootstraps write)")
    print(f"  Fresh data per epoch: {fresh_data}\n")

    optimizer = AdamW(all_params, lr=lr, weight_decay=0.01)

    tokenizer = surgical_model.tokenizer
    base_dataset = dataset
    base_formatted = [format_for_training(ex, tokenizer) for ex in dataset]

    for block in surgical_model.memory_blocks.values():
        block.train()

    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_c_loss = 0.0
        epoch_r_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

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

            # Collect keys per layer for contrastive loss
            batch_keys = {idx: [] for idx in surgical_model.layer_indices}
            accumulated_loss = None
            batch_lm_total = 0.0

            for idx_in_batch in batch_indices:
                ex = formatted[idx_in_batch]
                fact_text = dataset[idx_in_batch]["fact"]

                # Encode fact at all target layers (with grad)
                layer_results = encode_fact_with_grad(
                    surgical_model, fact_text, device
                )

                # Build per-layer memory banks and set them
                for layer_idx, (key, value, h_orig) in layer_results.items():
                    batch_keys[layer_idx].append(key)

                    bank = MemoryBank(
                        memory_dim=surgical_model.memory_blocks[layer_idx].memory_dim,
                        max_entries=256,
                    )
                    bank.write(key, value, detach=False)
                    surgical_model.set_memory_bank_for_layer(layer_idx, bank)

                # Forward pass (all hooks active with their respective banks)
                input_ids = ex["input_ids"].unsqueeze(0).to(device)
                labels = ex["labels"].unsqueeze(0).to(device)

                outputs = surgical_model(input_ids=input_ids, labels=labels)
                lm_loss = outputs.loss / len(batch_indices)
                batch_lm_total += outputs.loss.item()

                # Reconstruction loss — per layer
                r_loss_total = torch.tensor(0.0, device=device)
                for layer_idx, (key, value, h_orig) in layer_results.items():
                    block = surgical_model.memory_blocks[layer_idx]
                    h_recon = block.recon_proj(value)
                    r_loss_total = r_loss_total + F.mse_loss(h_recon, h_orig)
                    epoch_r_loss += F.mse_loss(h_recon, h_orig).item()

                r_loss = recon_weight * r_loss_total / (len(batch_indices) * len(surgical_model.layer_indices))

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

            # Contrastive loss — per layer, averaged
            if contrastive_weight > 0:
                c_loss_total = 0.0
                c_count = 0
                for layer_idx in surgical_model.layer_indices:
                    if len(batch_keys[layer_idx]) > 1:
                        keys_tensor = torch.stack(batch_keys[layer_idx])
                        c_loss = contrastive_key_loss(keys_tensor)
                        accumulated_loss = accumulated_loss + contrastive_weight * c_loss / len(surgical_model.layer_indices)
                        c_loss_total += c_loss.item()
                        c_count += 1
                if c_count > 0:
                    epoch_c_loss += c_loss_total / c_count

            accumulated_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            # Clamp gates on all memory blocks
            with torch.no_grad():
                for block in surgical_model.memory_blocks.values():
                    block.gate.clamp_(max=gate_max)

            epoch_loss += batch_lm_total

        avg_loss = epoch_loss / len(formatted)
        accuracy = epoch_correct / max(epoch_total, 1) * 100
        elapsed = time.time() - t0

        n_batches = (len(formatted) + batch_size - 1) // batch_size
        avg_c_loss = epoch_c_loss / max(n_batches, 1)
        avg_r_loss = epoch_r_loss / (len(formatted) * len(surgical_model.layer_indices))

        # Show gate values for all layers
        gate_strs = [
            f"L{idx}={surgical_model.memory_blocks[idx].gate.item():.4f}"
            for idx in surgical_model.layer_indices
        ]

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"r_loss={avg_r_loss:.4f}  "
            f"c_loss={avg_c_loss:.4f}  "
            f"acc={accuracy:.1f}%  "
            f"gates=[{', '.join(gate_strs)}]  "
            f"time={elapsed:.1f}s"
        )

        # Save checkpoint
        save_state = {
            "epoch": epoch + 1,
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
            "layer_indices": surgical_model.layer_indices,
        }
        for idx, block in surgical_model.memory_blocks.items():
            save_state[f"memory_block_{idx}"] = block.state_dict()
            save_state[f"gate_{idx}"] = block.gate.item()

        torch.save(save_state, checkpoint_dir / "latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_state, checkpoint_dir / "best.pt")
            print(f"  -> New best (loss={best_loss:.4f})")

        # Evaluate every 10 epochs
        if test_data and (epoch + 1) % 10 == 0:
            print(f"\n  --- Eval at epoch {epoch+1} (gate={eval_gate}) ---")
            saved_gates = {}
            for idx, block in surgical_model.memory_blocks.items():
                saved_gates[idx] = block.gate.data.clone()
                block.gate.data = torch.tensor(eval_gate)

            test_entries = pre_encode_facts(surgical_model, test_data, device)
            evaluate(surgical_model, test_data, test_entries, device=device)

            for idx, block in surgical_model.memory_blocks.items():
                block.gate.data = saved_gates[idx]
                block.train()
            print()

    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Memory Graft Joint Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=None,
                        help="Single layer (backward compat). Use --layers for multi-layer.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices for multi-layer injection. E.g. --layers 10 14 18")
    parser.add_argument("--memory_dim", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--checkpoint_dir", default="checkpoints/qwen7b_joint_v2")
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--recon_weight", type=float, default=0.5)
    parser.add_argument("--gate_max", type=float, default=0.4)
    parser.add_argument("--eval_gate", type=float, default=0.35)
    args = parser.parse_args()

    # Determine layer indices
    if args.layers is not None:
        layer_indices = args.layers
    elif args.layer is not None:
        layer_indices = [args.layer]
    else:
        layer_indices = [14]  # default single layer

    # Load model
    sm = SurgicalModel.from_pretrained(
        model_name=args.model,
        layer_indices=layer_indices,
        memory_dim=args.memory_dim,
        n_heads=args.n_heads,
        device=args.device,
    )

    d_model = sm.base_model.config.hidden_size
    print(f"\nMemory dim: {args.memory_dim} (bottleneck ratio: {d_model / args.memory_dim:.1f}x)")
    print(f"Layers: {layer_indices}")

    # Generate data
    print(f"Generating {args.n_train} train + {args.n_test} test examples...")
    train_data = generate_dataset(n=args.n_train, seed=42)
    test_data = generate_dataset(n=args.n_test, seed=99)

    # Train
    print("\n" + "=" * 60)
    print("JOINT TRAINING: Read + Write from scratch")
    print(f"  Layers: {layer_indices}")
    print(f"  Memory dim: {args.memory_dim}")
    print("=" * 60)

    train_joint(
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
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION (gate={args.eval_gate})")
    print("=" * 60)
    for block in sm.memory_blocks.values():
        block.gate.data = torch.tensor(args.eval_gate)
    test_entries = pre_encode_facts(sm, test_data, args.device)
    evaluate(sm, test_data, test_entries, device=args.device)

    sm.save_memory_block(f"{args.checkpoint_dir}/memory_block_final.pt")
    print("\nJoint training complete.")


if __name__ == "__main__":
    main()
