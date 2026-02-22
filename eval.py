import torch
from mvp.surgery import SurgicalModel
from mvp.train import pre_encode_facts, evaluate
from mvp.data import generate_dataset

sm = SurgicalModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    layer_indices=[10, 14, 18],
    memory_dim=1024,
    device="cuda",
)
sm.load_memory_block("checkpoints/qwen7b_joint_v2/best.pt")

for gate_val in [0.15, 0.20, 0.25, 0.30]:
    print(f"\n{'='*60}")
    print(f"GATE = {gate_val}")
    print(f"{'='*60}")
    for block in sm.memory_blocks.values():
        block.gate.data = torch.tensor(gate_val)
    test_data = generate_dataset(n=50, seed=99)
    entries = pre_encode_facts(sm, test_data, "cuda")
    evaluate(sm, test_data, entries, device="cuda")
