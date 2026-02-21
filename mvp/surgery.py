"""
Model Surgery — Load a frozen transformer, splice in MemoryBlocks.

SurgicalModel wraps a HuggingFace model and one or more MemoryBlocks.
It installs forward hooks at the target layers that route hidden states
through the memory blocks during the forward pass.

The base model's weights NEVER change. Only the memory blocks are trainable.

Supports multi-layer injection: separate MemoryBlock per layer, each with
its own read/write pathways, hooks, and capture buffers.
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memory_block import MemoryBlock
from .memory_bank import MemoryBank


class SurgicalModel:
    """
    Wraps a frozen base model + trainable memory blocks (one per target layer).

    Usage:
        # Single layer (backward compatible):
        sm = SurgicalModel.from_pretrained("...", layer_indices=[14])

        # Multi-layer:
        sm = SurgicalModel.from_pretrained("...", layer_indices=[10, 14, 18])

        sm.set_memory_bank(my_bank)
        outputs = sm(input_ids=ids, labels=labels)
    """

    def __init__(self, base_model, tokenizer, memory_blocks, layer_indices, device):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.layer_indices = layer_indices
        self.device = device

        # Dict: layer_idx -> MemoryBlock
        self.memory_blocks = {idx: block for idx, block in zip(layer_indices, memory_blocks)}

        # Per-layer state
        self.memory_banks = {idx: None for idx in layer_indices}
        self._memory_active = True
        self._capture_buffers = {idx: None for idx in layer_indices}
        self._capture_mode = False

        # Install hooks — one per target layer
        self._hook_handles = {}
        for idx in layer_indices:
            layer = self.base_model.model.layers[idx]
            self._hook_handles[idx] = layer.register_forward_hook(
                self._make_hook(idx)
            )

        # Backward compat: expose first memory block as .memory_block
        self.memory_block = memory_blocks[0]

    def _make_hook(self, layer_idx):
        """Create a hook closure for a specific layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Capture mode — grab hidden states for encoding facts
            if self._capture_mode:
                self._capture_buffers[layer_idx] = hidden_states.detach()

            # Memory mode — apply memory block for this layer
            bank = self.memory_banks.get(layer_idx)
            block = self.memory_blocks[layer_idx]

            if self._memory_active and bank is not None and bank.has_entries():
                orig_dtype = hidden_states.dtype
                needs_squeeze = False
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    needs_squeeze = True

                modified = block.read(
                    hidden_states.float(), bank, device=self.device
                ).to(orig_dtype)

                if needs_squeeze:
                    modified = modified.squeeze(0)

                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                else:
                    return modified

            return output
        return hook

    def set_memory_bank(self, memory_bank):
        """Set the same memory bank for all layers."""
        for idx in self.layer_indices:
            self.memory_banks[idx] = memory_bank

    def set_memory_bank_for_layer(self, layer_idx, memory_bank):
        """Set a memory bank for a specific layer."""
        self.memory_banks[layer_idx] = memory_bank

    def encode_fact(self, text, memory_bank):
        """
        Encode a fact into the memory bank using the FIRST layer's write pathway.

        For multi-layer: captures hidden states at all target layers,
        but writes using the first layer's memory block (for backward compat).
        Use encode_fact_multi for per-layer encoding.
        """
        self._memory_active = False
        self._capture_mode = True
        for idx in self.layer_indices:
            self._capture_buffers[idx] = None

        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).input_ids.to(self.device)

        with torch.no_grad():
            self.base_model(input_ids)

        self._capture_mode = False
        self._memory_active = True

        first_idx = self.layer_indices[0]
        if self._capture_buffers[first_idx] is None:
            raise RuntimeError("Failed to capture hidden states — hook didn't fire")

        h = self._capture_buffers[first_idx].to(self.device)
        if h.dim() == 2:
            h = h.unsqueeze(0)

        block = self.memory_blocks[first_idx]
        with torch.no_grad():
            keys, values = block.encode_to_memory(h.float())
            memory_bank.write(keys.squeeze(0), values.squeeze(0))

    def encode_fact_multi(self, text):
        """
        Encode a fact at ALL target layers.

        Returns: dict of {layer_idx: MemoryBank} — one bank per layer,
        each containing the fact encoded through that layer's write pathway.
        """
        self._memory_active = False
        self._capture_mode = True
        for idx in self.layer_indices:
            self._capture_buffers[idx] = None

        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).input_ids.to(self.device)

        with torch.no_grad():
            self.base_model(input_ids)

        self._capture_mode = False
        self._memory_active = True

        banks = {}
        for idx in self.layer_indices:
            if self._capture_buffers[idx] is None:
                raise RuntimeError(f"Failed to capture hidden states at layer {idx}")

            h = self._capture_buffers[idx].to(self.device)
            if h.dim() == 2:
                h = h.unsqueeze(0)

            block = self.memory_blocks[idx]
            bank = MemoryBank(memory_dim=block.memory_dim, max_entries=256)

            with torch.no_grad():
                keys, values = block.encode_to_memory(h.float())
                bank.write(keys.squeeze(0), values.squeeze(0))

            banks[idx] = bank

        return banks

    def __call__(self, **kwargs):
        """Forward pass through the base model (with memory hooks active)."""
        return self.base_model(**kwargs)

    def generate(self, **kwargs):
        """Generate text (with memory hooks active)."""
        return self.base_model.generate(**kwargs)

    def trainable_parameters(self):
        """Return all memory blocks' parameters (for optimizer)."""
        params = []
        for block in self.memory_blocks.values():
            params.extend(block.parameters())
        return params

    def all_memory_blocks(self):
        """Return list of (layer_idx, memory_block) tuples."""
        return [(idx, self.memory_blocks[idx]) for idx in self.layer_indices]

    @classmethod
    def from_pretrained(
        cls,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        layer_idx=None,
        layer_indices=None,
        memory_dim=512,
        n_heads=8,
        device="cuda",
        dtype=torch.float16,
    ):
        """
        Load a pretrained model, freeze it, create memory blocks, splice them in.

        Args:
            layer_idx: Single layer (backward compat). Ignored if layer_indices is set.
            layer_indices: List of layers to inject memory at. E.g. [10, 14, 18].
        """
        # Handle backward compat: single layer_idx → list
        if layer_indices is None:
            if layer_idx is not None:
                layer_indices = [layer_idx]
            else:
                layer_indices = [11]  # default

        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            base_model = base_model.to(device)

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        frozen_count = sum(p.numel() for p in base_model.parameters())
        print(f"Frozen {frozen_count / 1e9:.2f}B base model parameters")

        d_model = base_model.config.hidden_size
        n_layers = base_model.config.num_hidden_layers

        for idx in layer_indices:
            if idx >= n_layers:
                raise ValueError(
                    f"layer_idx={idx} but model only has {n_layers} layers"
                )

        # Create one memory block per target layer
        memory_blocks = []
        total_mem_params = 0
        for idx in layer_indices:
            block = MemoryBlock(
                d_model=d_model,
                memory_dim=memory_dim,
                n_heads=n_heads,
            ).to(device).float()
            memory_blocks.append(block)
            total_mem_params += block.param_count()

        if total_mem_params >= 1_000_000:
            param_str = f"{total_mem_params / 1_000_000:.1f}M"
        else:
            param_str = f"{total_mem_params / 1_000:.1f}K"

        print(f"Memory blocks: {param_str} trainable parameters across {len(layer_indices)} layers")
        print(f"Spliced at layers {layer_indices} / {n_layers}")

        return cls(base_model, tokenizer, memory_blocks, layer_indices, device)

    def save_memory_block(self, path):
        """Save all memory block weights."""
        state = {}
        for idx, block in self.memory_blocks.items():
            state[f"layer_{idx}"] = block.state_dict()
        torch.save(state, path)
        print(f"Saved {len(self.memory_blocks)} memory block(s) to {path}")

    def load_memory_block(self, path):
        """Load memory block weights."""
        state = torch.load(path, weights_only=True, map_location=self.device)

        # Handle old single-block checkpoints
        if all(not k.startswith("layer_") for k in state.keys()):
            first_idx = self.layer_indices[0]
            self.memory_blocks[first_idx].load_state_dict(state)
            print(f"Loaded single memory block from {path} into layer {first_idx}")
            return

        for key, block_state in state.items():
            idx = int(key.split("_")[1])
            if idx in self.memory_blocks:
                self.memory_blocks[idx].load_state_dict(block_state)
        print(f"Loaded memory blocks from {path}")
