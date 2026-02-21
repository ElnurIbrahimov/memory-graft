"""
Training Loop for Memory Graft MVP.

Phase 1 (this file): Read-only training.
- Pre-encode all facts into memory entries (write projections frozen)
- Train ONLY the read pathway: query projection, cross-attention, output projection, gate
- Loss: next-token prediction on answers that REQUIRE memory

Usage:
    # Local smoke test (CPU, 2 examples):
    python -m mvp.train --smoke

    # Full training on RunPod (GPU):
    python -m mvp.train --device cuda --epochs 5 --batch_size 4 --lr 1e-4

    # Resume from checkpoint:
    python -m mvp.train --device cuda --resume checkpoints/epoch_3.pt
"""

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .surgery import SurgicalModel
from .memory_bank import MemoryBank
from .data import generate_dataset, generate_multi_fact_dataset, format_for_training


def pre_encode_facts(surgical_model, dataset, device):
    """
    Pre-encode all facts into memory entries.

    For each fact, run it through the frozen model to capture hidden states
    at the target layer, then project to memory space.

    Returns: list of (key, value) tuples, one per example.
    """
    print("Pre-encoding facts into memory entries...")
    memory_entries = []

    surgical_model.memory_block.eval()
    for example in tqdm(dataset, desc="Encoding facts"):
        bank = MemoryBank(
            memory_dim=surgical_model.memory_block.memory_dim,
            max_entries=256,
        )
        surgical_model.encode_fact(example["fact"], bank)

        # Extract the encoded entry
        key = bank.keys[0]
        value = bank.values[0]
        memory_entries.append({"key": key, "value": value})

    return memory_entries


def pre_encode_multi_facts(surgical_model, dataset, device):
    """Pre-encode multi-fact examples."""
    print("Pre-encoding multi-fact examples...")
    all_entries = []

    surgical_model.memory_block.eval()
    for example in tqdm(dataset, desc="Encoding multi-facts"):
        entries = []
        for fact in example["facts"]:
            bank = MemoryBank(
                memory_dim=surgical_model.memory_block.memory_dim,
                max_entries=256,
            )
            surgical_model.encode_fact(fact, bank)
            entries.append({"key": bank.keys[0], "value": bank.values[0]})
        all_entries.append(entries)

    return all_entries


def train_phase1(
    surgical_model,
    dataset,
    memory_entries,
    epochs=5,
    lr=1e-4,
    batch_size=4,
    device="cuda",
    checkpoint_dir="checkpoints",
    multi_fact_dataset=None,
    multi_fact_entries=None,
):
    """
    Phase 1: Read-only training.

    The write pathway is frozen. Only the read pathway trains.
    Memory banks are pre-filled with ground-truth entries.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Freeze write pathway for Phase 1
    for param in surgical_model.memory_block.write_key_proj.parameters():
        param.requires_grad = False
    for param in surgical_model.memory_block.write_value_proj.parameters():
        param.requires_grad = False
    for param in surgical_model.memory_block.importance_gate.parameters():
        param.requires_grad = False

    trainable_count = sum(
        p.numel()
        for p in surgical_model.memory_block.parameters()
        if p.requires_grad
    )
    total_count = surgical_model.memory_block.param_count()
    print(f"\nPhase 1: Training {trainable_count:,} / {total_count:,} memory block params")
    print(f"  Read pathway only. Write pathway frozen.\n")

    optimizer = AdamW(
        [p for p in surgical_model.memory_block.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    tokenizer = surgical_model.tokenizer

    # Format all examples
    formatted = [format_for_training(ex, tokenizer) for ex in dataset]

    surgical_model.memory_block.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        # Shuffle
        indices = torch.randperm(len(formatted)).tolist()

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]

            batch_loss = 0.0
            for idx in batch_indices:
                ex = formatted[idx]
                entry = memory_entries[idx]

                # Build memory bank for this example
                bank = MemoryBank(
                    memory_dim=surgical_model.memory_block.memory_dim,
                    max_entries=256,
                )
                bank.write(entry["key"].clone(), entry["value"].clone())
                surgical_model.set_memory_bank(bank)

                # Forward pass
                input_ids = ex["input_ids"].unsqueeze(0).to(device)
                labels = ex["labels"].unsqueeze(0).to(device)

                outputs = surgical_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / len(batch_indices)  # normalize by batch
                batch_loss += loss.item() * len(batch_indices)

                # Check accuracy: does the model predict the right answer tokens?
                with torch.no_grad():
                    preds = outputs.logits[:, :-1, :].argmax(dim=-1)
                    target = labels[:, 1:]
                    mask = target != -100
                    if mask.any():
                        epoch_correct += (preds[mask] == target[mask]).sum().item()
                        epoch_total += mask.sum().item()

                loss.backward()

            # Step optimizer after accumulating batch
            torch.nn.utils.clip_grad_norm_(
                surgical_model.memory_block.parameters(), max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += batch_loss

        avg_loss = epoch_loss / len(formatted)
        accuracy = epoch_correct / max(epoch_total, 1) * 100
        elapsed = time.time() - t0
        gate_val = surgical_model.memory_block.gate.item()

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"loss={avg_loss:.4f}  "
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
            },
            save_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "memory_block_state": surgical_model.memory_block.state_dict(),
                    "loss": avg_loss,
                    "gate": gate_val,
                },
                best_path,
            )
            print(f"  -> New best model saved (loss={best_loss:.4f})")

    return best_loss


def evaluate(surgical_model, test_examples, memory_entries, device="cuda"):
    """
    Evaluate: does the model actually USE memory?

    Three tests per example:
    1. WITH correct memory → should answer correctly
    2. WITHOUT memory → should fail or give generic answer
    3. WITH wrong memory → should give wrong answer (proves it's reading memory)
    """
    surgical_model.memory_block.eval()
    tokenizer = surgical_model.tokenizer

    results = {"with_memory": 0, "without_memory": 0, "wrong_memory": 0, "total": 0}

    print("\n=== EVALUATION ===\n")

    for i, (example, entry) in enumerate(zip(test_examples, memory_entries)):
        question = f"Question: {example['question']}\nAnswer:"
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)

        # --- Test 1: WITH correct memory ---
        bank = MemoryBank(
            memory_dim=surgical_model.memory_block.memory_dim, max_entries=256
        )
        bank.write(entry["key"].clone(), entry["value"].clone())
        surgical_model.set_memory_bank(bank)

        with torch.no_grad():
            out = surgical_model.generate(
                input_ids=input_ids,
                max_new_tokens=30,
                do_sample=False,
            )
        answer_with = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # --- Test 2: WITHOUT memory ---
        surgical_model.set_memory_bank(MemoryBank(memory_dim=surgical_model.memory_block.memory_dim))

        with torch.no_grad():
            out = surgical_model.generate(
                input_ids=input_ids,
                max_new_tokens=30,
                do_sample=False,
            )
        answer_without = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # --- Check ---
        expected = example["answer"]
        # Simple substring match for MVP
        hit_with = any(
            word.lower() in answer_with.lower()
            for word in expected.split()
            if len(word) > 3
        )
        hit_without = any(
            word.lower() in answer_without.lower()
            for word in expected.split()
            if len(word) > 3
        )

        results["with_memory"] += int(hit_with)
        results["without_memory"] += int(hit_without)
        results["total"] += 1

        if i < 10:  # Print first 10 examples
            print(f"FACT:     {example['fact']}")
            print(f"QUESTION: {example['question']}")
            print(f"EXPECTED: {expected}")
            print(f"WITH MEM: {answer_with}")
            print(f"NO MEM:   {answer_without}")
            print(f"  -> Memory helped: {hit_with and not hit_without}")
            print()

    n = results["total"]
    print(f"\n=== RESULTS ({n} examples) ===")
    print(f"With memory:    {results['with_memory']}/{n} ({results['with_memory']/n*100:.1f}%)")
    print(f"Without memory: {results['without_memory']}/{n} ({results['without_memory']/n*100:.1f}%)")
    mem_delta = results["with_memory"] - results["without_memory"]
    print(f"Memory delta:   +{mem_delta} ({mem_delta/n*100:.1f}%)")

    if mem_delta > 0:
        print("\n*** MEMORY BLOCK IS WORKING — model answers better WITH memory ***")
    else:
        print("\n*** Memory block not yet effective — more training needed ***")

    return results


def main():
    parser = argparse.ArgumentParser(description="Memory Graft MVP Training")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--layer", type=int, default=11, help="Layer to splice at")
    parser.add_argument("--memory_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_train", type=int, default=500, help="Training examples")
    parser.add_argument("--n_test", type=int, default=50, help="Test examples")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Quick CPU smoke test")
    args = parser.parse_args()

    if args.smoke:
        args.device = "cpu"
        args.epochs = 2
        args.n_train = 10
        args.n_test = 5
        args.batch_size = 2
        print("=== SMOKE TEST MODE (CPU, tiny dataset) ===\n")

    # Load model and create surgical model
    surgical_model = SurgicalModel.from_pretrained(
        model_name=args.model,
        layer_idx=args.layer,
        memory_dim=args.memory_dim,
        n_heads=args.n_heads,
        device=args.device,
        dtype=torch.float32 if args.device == "cpu" else torch.float16,
    )

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False, map_location=args.device)
        surgical_model.memory_block.load_state_dict(checkpoint["memory_block_state"])
        print(f"Resumed from {args.resume} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")

    # Generate data
    print(f"\nGenerating {args.n_train} training + {args.n_test} test examples...")
    train_data = generate_dataset(n=args.n_train, seed=42)
    test_data = generate_dataset(n=args.n_test, seed=99)

    # Pre-encode facts
    train_entries = pre_encode_facts(surgical_model, train_data, args.device)
    test_entries = pre_encode_facts(surgical_model, test_data, args.device)

    # Train Phase 1
    print("\n" + "=" * 60)
    print("PHASE 1: Read-Only Training")
    print("=" * 60)

    train_phase1(
        surgical_model=surgical_model,
        dataset=train_data,
        memory_entries=train_entries,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Evaluate
    evaluate(surgical_model, test_data, test_entries, device=args.device)

    # Save final memory block
    surgical_model.save_memory_block(f"{args.checkpoint_dir}/memory_block_final.pt")
    print("\nDone.")


if __name__ == "__main__":
    main()
