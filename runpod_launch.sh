#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== MEMORY GRAFT - H100 ==="
echo ""
echo "Step 1: Dependencies..."
pip install -q transformers accelerate safetensors tqdm
echo ""
echo "Step 2: Smoke test..."
python -m mvp.smoke_test
echo ""
echo "Step 3: TinyLlama validation..."
python -m mvp.train --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --layer 11 --device cuda --epochs 5 --batch_size 8 --lr 1e-4 --n_train 200 --n_test 30 --checkpoint_dir checkpoints/tinyllama
echo ""
echo "=== TinyLlama done. Starting Qwen... ==="
echo ""
echo "Step 4: Qwen 7B training..."
python -m mvp.train --model Qwen/Qwen2.5-7B-Instruct --layer 14 --device cuda --epochs 10 --batch_size 4 --lr 1e-4 --n_train 500 --n_test 50 --checkpoint_dir checkpoints/qwen7b
echo ""
echo "=== ALL DONE ==="
echo "Download checkpoints/ before stopping pod!"
