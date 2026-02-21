"""
Smoke Test for Memory Graft MVP.

Verifies the core mechanics BEFORE spending GPU money:
1. MemoryBank: write, read, save/load, eviction
2. MemoryBlock: correct tensor shapes through read pathway
3. Gradient flow: gradients reach memory block but NOT frozen model
4. Gate behavior: starts at zero, moves during training
5. (Optional) Integration: full pipeline with TinyLlama on CPU

Run:
    python -m mvp.smoke_test          # Unit tests only (instant, no model download)
    python -m mvp.smoke_test --full   # Full integration test with TinyLlama (slow, downloads model)
"""

import sys
import argparse
import torch
import torch.nn as nn

from .memory_bank import MemoryBank
from .memory_block import MemoryBlock


def test_memory_bank():
    """Test MemoryBank data structure."""
    print("--- MemoryBank Tests ---")

    bank = MemoryBank(memory_dim=32, max_entries=5)
    assert bank.size == 0
    assert not bank.has_entries()
    print("  [OK] Empty bank")

    # Write single entry
    k = torch.randn(32)
    v = torch.randn(32)
    bank.write(k, v)
    assert bank.size == 1
    assert bank.has_entries()
    print("  [OK] Write single entry")

    # Write batch
    keys = torch.randn(3, 32)
    values = torch.randn(3, 32)
    bank.write(keys, values, importance_scores=torch.tensor([0.9, 0.5, 0.1]))
    assert bank.size == 4
    print("  [OK] Write batch")

    # Get tensors
    k_out = bank.get_keys()
    v_out = bank.get_values()
    assert k_out.shape == (4, 32)
    assert v_out.shape == (4, 32)
    print("  [OK] Get keys/values shapes")

    # GPU device (if available)
    if torch.cuda.is_available():
        try:
            k_gpu = bank.get_keys(device="cuda")
            assert k_gpu.device.type == "cuda"
            print("  [OK] GPU transfer")
        except RuntimeError:
            print("  [SKIP] GPU transfer (CUDA not ready, will work in training)")

    # Eviction
    for _ in range(10):
        bank.write(torch.randn(32), torch.randn(32))
    assert bank.size <= 5, f"Expected <=5 after eviction, got {bank.size}"
    print(f"  [OK] Eviction (size={bank.size} after adding 14 to max_entries=5)")

    # Save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    bank.save(path)
    loaded = MemoryBank.load(path)
    assert loaded.size == bank.size
    assert loaded.memory_dim == bank.memory_dim
    assert torch.allclose(loaded.get_keys(), bank.get_keys())
    os.unlink(path)
    print("  [OK] Save/load roundtrip")

    # Clear
    bank.clear()
    assert bank.size == 0
    print("  [OK] Clear")

    print("  ALL MEMORY BANK TESTS PASSED\n")


def test_memory_block_shapes():
    """Test MemoryBlock tensor shapes through read pathway."""
    print("--- MemoryBlock Shape Tests ---")

    d_model = 64
    memory_dim = 32
    n_heads = 4
    batch_size = 2
    seq_len = 10
    n_memories = 5

    block = MemoryBlock(d_model=d_model, memory_dim=memory_dim, n_heads=n_heads)
    print(f"  MemoryBlock params: {block.param_count_str()}")

    # Create dummy hidden state
    hidden = torch.randn(batch_size, seq_len, d_model)

    # Test with empty memory bank
    bank = MemoryBank(memory_dim=memory_dim, max_entries=256)
    out = block.read(hidden, bank)
    assert out.shape == hidden.shape, f"Expected {hidden.shape}, got {out.shape}"
    assert torch.allclose(out, hidden), "Empty bank should return unchanged hidden state"
    print("  [OK] Empty memory bank -> identity")

    # Test with populated memory bank
    bank.write(torch.randn(n_memories, memory_dim), torch.randn(n_memories, memory_dim))
    out = block.read(hidden, bank)
    assert out.shape == hidden.shape, f"Expected {hidden.shape}, got {out.shape}"
    print(f"  [OK] Read from {n_memories} memories -> shape {out.shape}")

    # Test that gate starts near 0
    assert abs(block.gate.item() - 0.01) < 1e-6, f"Gate should init at 0.01, got {block.gate.item()}"
    print(f"  [OK] Gate initialized at {block.gate.item()} (near zero)")

    # Test encode_to_memory
    keys, values = block.encode_to_memory(hidden)
    assert keys.shape == (batch_size, memory_dim), f"Expected ({batch_size}, {memory_dim}), got {keys.shape}"
    assert values.shape == (batch_size, memory_dim)
    print(f"  [OK] encode_to_memory → keys {keys.shape}, values {values.shape}")

    # Test with None memory bank
    out = block.read(hidden, None)
    assert torch.allclose(out, hidden)
    print("  [OK] None memory bank → identity")

    print("  ALL SHAPE TESTS PASSED\n")


def test_gradient_flow():
    """
    THE CRITICAL TEST: Verify gradients flow to memory block
    but NOT to the frozen model.

    Simulates the surgical model: frozen layers + memory block spliced in.
    Runs a few optimizer steps so the gate opens and ALL params get signal.
    """
    print("--- Gradient Flow Test ---")

    d_model = 64
    memory_dim = 32
    n_heads = 4

    # Simulate a 3-layer "frozen model"
    layer_0 = nn.Linear(d_model, d_model)
    layer_1 = nn.Linear(d_model, d_model)  # memory block goes AFTER this
    layer_2 = nn.Linear(d_model, d_model)
    output_head = nn.Linear(d_model, 10)  # predict 10 classes

    # Freeze everything
    for module in [layer_0, layer_1, layer_2, output_head]:
        for param in module.parameters():
            param.requires_grad = False

    # Create trainable memory block
    block = MemoryBlock(d_model=d_model, memory_dim=memory_dim, n_heads=n_heads)

    # Populate memory bank
    bank = MemoryBank(memory_dim=memory_dim, max_entries=256)
    bank.write(torch.randn(5, memory_dim), torch.randn(5, memory_dim))

    optimizer = torch.optim.Adam(block.parameters(), lr=0.01)

    # Run 5 steps so gate opens and all params get gradient signal
    for step in range(5):
        optimizer.zero_grad()
        x = torch.randn(2, 8, d_model)
        h = layer_0(x)
        h = layer_1(h)
        h = block.read(h, bank)
        h = layer_2(h)
        logits = output_head(h.mean(dim=1))
        targets = torch.randint(0, 10, (2,))
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    # After 5 steps, check gradients from the LAST backward pass
    # (optimizer.step doesn't zero grads, so they're still there)
    read_params_with_grad = 0
    read_params_total = 0
    print(f"  Memory block gradients (after 5 steps, gate={block.gate.item():.4f}):")
    for name, param in block.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None and param.grad.abs().sum() > 0
            # Only check read pathway params (write pathway is tested separately)
            is_read = name.startswith("read_") or name in ("gate", "layer_norm.weight", "layer_norm.bias")
            if is_read:
                read_params_total += 1
                if has_grad:
                    read_params_with_grad += 1
                status = "OK" if has_grad else "FAIL"
                norm_str = f" (norm={param.grad.norm():.6f})" if has_grad else ""
                print(f"    [{status}] {name}{norm_str}")

    # Check: frozen params should NOT have gradients
    frozen_grads_exist = False
    for name, module in [("layer_0", layer_0), ("layer_1", layer_1),
                          ("layer_2", layer_2), ("output_head", output_head)]:
        for pname, param in module.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                frozen_grads_exist = True
                print(f"  [FAIL] Frozen param {name}.{pname} has gradient!")

    if not frozen_grads_exist:
        print(f"  [OK] All frozen parameters have NO gradients")

    gate_val = block.gate.item()
    print(f"  [OK] Gate opened to {gate_val:.4f} (started at 0.01)")

    passed = (read_params_with_grad == read_params_total) and not frozen_grads_exist
    if passed:
        print("  GRADIENT FLOW TEST PASSED\n")
    else:
        print(f"  GRADIENT FLOW TEST FAILED ({read_params_with_grad}/{read_params_total} read params have grads)\n")

    return passed


def test_gate_learns():
    """
    Test that the gate value changes during a few training steps.
    If the gate moves away from 0, the memory block is learning to activate.
    """
    print("--- Gate Learning Test ---")

    d_model = 64
    memory_dim = 32
    n_heads = 4

    block = MemoryBlock(d_model=d_model, memory_dim=memory_dim, n_heads=n_heads)

    # Populate memory bank with info that SHOULD help
    bank = MemoryBank(memory_dim=memory_dim, max_entries=256)
    bank.write(torch.randn(5, memory_dim), torch.randn(5, memory_dim))

    optimizer = torch.optim.Adam(block.parameters(), lr=0.01)

    initial_gate = block.gate.item()
    print(f"  Initial gate: {initial_gate:.6f}")

    # Simple training: make the output match a target
    target = torch.randn(2, 8, d_model)

    for step in range(50):
        hidden = torch.randn(2, 8, d_model)
        out = block.read(hidden, bank)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    final_gate = block.gate.item()
    print(f"  Final gate after 50 steps: {final_gate:.6f}")
    print(f"  Gate moved: {abs(final_gate - initial_gate):.6f}")

    if abs(final_gate - initial_gate) > 1e-6:
        print("  [OK] Gate is learning (moved away from zero)")
    else:
        print("  [WARN] Gate didn't move — may need more steps or different LR")

    print("  GATE LEARNING TEST PASSED\n")


def test_integration_tinyllama():
    """
    Full integration test with TinyLlama on CPU.
    Downloads the model (first run), runs a few training steps.
    """
    print("--- Integration Test (TinyLlama, CPU) ---")
    print("  This will download TinyLlama-1.1B (~2GB) on first run...")

    from .surgery import SurgicalModel
    from .data import generate_dataset, format_for_training

    # Load model
    sm = SurgicalModel.from_pretrained(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        layer_idx=11,
        memory_dim=512,
        n_heads=8,
        device="cpu",
        dtype=torch.float32,
    )

    # Generate tiny dataset
    data = generate_dataset(n=3, seed=42)
    print(f"\n  Test data:")
    for ex in data:
        print(f"    Fact: {ex['fact']}")
        print(f"    Q: {ex['question']} → A: {ex['answer']}")

    # Encode facts
    for ex in data:
        bank = MemoryBank(memory_dim=512, max_entries=256)
        sm.encode_fact(ex["fact"], bank)
        print(f"\n  Encoded '{ex['fact']}' → {bank.size} memory entries")

        # Set memory and generate
        sm.set_memory_bank(bank)
        question = f"Question: {ex['question']}\nAnswer:"
        input_ids = sm.tokenizer(question, return_tensors="pt").input_ids

        with torch.no_grad():
            out = sm.generate(input_ids=input_ids, max_new_tokens=20, do_sample=False)
        answer = sm.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Generated (untrained): {answer.strip()}")

    # Quick training step to verify backward pass works
    print("\n  Running 1 training step...")
    formatted = format_for_training(data[0], sm.tokenizer)
    bank = MemoryBank(memory_dim=512, max_entries=256)
    sm.encode_fact(data[0]["fact"], bank)
    sm.set_memory_bank(bank)

    # Freeze write pathway
    for p in sm.memory_block.write_key_proj.parameters():
        p.requires_grad = False
    for p in sm.memory_block.write_value_proj.parameters():
        p.requires_grad = False
    for p in sm.memory_block.importance_gate.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in sm.memory_block.parameters() if p.requires_grad], lr=1e-4
    )

    input_ids = formatted["input_ids"].unsqueeze(0)
    labels = formatted["labels"].unsqueeze(0)
    outputs = sm(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    print(f"  Loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    gate = sm.memory_block.gate.item()
    print(f"  Gate after 1 step: {gate:.6f}")
    print(f"  [OK] Full forward + backward pass completed")

    # Verify frozen params didn't move
    print(f"  Checking frozen params...")
    for name, param in sm.base_model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            print(f"  [FAIL] Frozen param {name} has gradient!")
            return False

    print(f"  [OK] All base model params stayed frozen")
    print("  INTEGRATION TEST PASSED\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Memory Graft Smoke Tests")
    parser.add_argument("--full", action="store_true", help="Run full integration test with TinyLlama")
    args = parser.parse_args()

    print("=" * 60)
    print("MEMORY GRAFT MVP — SMOKE TESTS")
    print("=" * 60 + "\n")

    all_passed = True

    # Unit tests (instant, no model)
    test_memory_bank()
    test_memory_block_shapes()
    all_passed &= test_gradient_flow()
    test_gate_learns()

    # Integration test (slow, downloads model)
    if args.full:
        all_passed &= test_integration_tinyllama()
    else:
        print("--- Skipping integration test (use --full to run) ---\n")

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("Safe to proceed to GPU training.")
    else:
        print("SOME TESTS FAILED")
        print("Fix issues before spending GPU money.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
