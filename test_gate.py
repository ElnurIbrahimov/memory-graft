import torch
from mvp.surgery import SurgicalModel
from mvp.data import generate_dataset
from mvp.train import pre_encode_facts, evaluate

sm = SurgicalModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", layer_idx=14, device="cuda")
best = torch.load("checkpoints/qwen7b_v2/best.pt", map_location="cuda", weights_only=False)
sm.memory_block.load_state_dict(best["memory_block_state"])
print(f"Loaded: gate={best['gate']:.4f}")

test_data = generate_dataset(n=50, seed=99)
test_entries = pre_encode_facts(sm, test_data, "cuda")

for g in [0.01, 0.03, 0.05, 0.1, 0.2, 0.35]:
    sm.memory_block.gate.data = torch.tensor(g)
    print(f"\n===== GATE = {g} =====")
    evaluate(sm, test_data[:10], test_entries[:10], device="cuda")
