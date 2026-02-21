"""
Model Surgery — Load a frozen transformer, splice in a MemoryBlock.

SurgicalModel wraps a HuggingFace model and a MemoryBlock.
It installs a forward hook at the target layer that routes hidden states
through the memory block during the forward pass.

The base model's weights NEVER change. Only the memory block is trainable.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memory_block import MemoryBlock
from .memory_bank import MemoryBank


class SurgicalModel:
    """
    Wraps a frozen base model + trainable memory block.

    Usage:
        sm = SurgicalModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", layer_idx=11)
        sm.set_memory_bank(my_bank)
        outputs = sm(input_ids=ids, labels=labels)  # loss backprops to memory block only
    """

    def __init__(self, base_model, tokenizer, memory_block, layer_idx, device):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.memory_block = memory_block
        self.layer_idx = layer_idx
        self.device = device

        self.memory_bank = None
        self._memory_active = True
        self._capture_buffer = None
        self._capture_mode = False

        # Install the hook
        self._hook_handle = self._target_layer().register_forward_hook(self._hook)

    def _target_layer(self):
        return self.base_model.model.layers[self.layer_idx]

    def _hook(self, module, input, output):
        """
        Forward hook on the target decoder layer.

        Two modes:
        1. Capture mode: store hidden states for memory encoding (no modification)
        2. Memory mode: route hidden states through memory block
        """
        # Handle both tuple and non-tuple outputs from decoder layer
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Capture mode — grab hidden states for encoding facts into memory
        if self._capture_mode:
            self._capture_buffer = hidden_states.detach()

        # Memory mode — apply memory block
        if (
            self._memory_active
            and self.memory_bank is not None
            and self.memory_bank.has_entries()
        ):
            # Ensure 3D for memory block
            needs_squeeze = False
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
                needs_squeeze = True

            modified = self.memory_block.read(
                hidden_states, self.memory_bank, device=self.device
            )

            if needs_squeeze:
                modified = modified.squeeze(0)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            else:
                return modified

        return output

    def set_memory_bank(self, memory_bank):
        """Set the active memory bank for inference/training."""
        self.memory_bank = memory_bank

    def encode_fact(self, text, memory_bank):
        """
        Encode a fact into the memory bank.

        1. Tokenize the text
        2. Run through frozen model up to target layer (capture mode)
        3. Mean-pool hidden states
        4. Project through write pathway → store in memory bank
        """
        self._memory_active = False
        self._capture_mode = True
        self._capture_buffer = None

        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).input_ids.to(self.device)

        with torch.no_grad():
            self.base_model(input_ids)

        self._capture_mode = False
        self._memory_active = True

        if self._capture_buffer is None:
            raise RuntimeError("Failed to capture hidden states — hook didn't fire")

        # Project captured hidden states to memory space
        h = self._capture_buffer.to(self.device)

        # Ensure 3D: [batch, seq_len, d_model]
        if h.dim() == 2:
            h = h.unsqueeze(0)

        with torch.no_grad():
            keys, values = self.memory_block.encode_to_memory(h)
            memory_bank.write(keys.squeeze(0), values.squeeze(0))

    def __call__(self, **kwargs):
        """Forward pass through the base model (with memory hook active)."""
        return self.base_model(**kwargs)

    def generate(self, **kwargs):
        """Generate text (with memory hook active)."""
        return self.base_model.generate(**kwargs)

    def trainable_parameters(self):
        """Return only the memory block's parameters (for optimizer)."""
        return self.memory_block.parameters()

    @classmethod
    def from_pretrained(
        cls,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        layer_idx=11,
        memory_dim=512,
        n_heads=8,
        device="cuda",
        dtype=torch.float16,
    ):
        """
        Load a pretrained model, freeze it, create a memory block, splice it in.

        Returns a ready-to-use SurgicalModel.
        """
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

        # Get model dimension from config
        d_model = base_model.config.hidden_size
        n_layers = base_model.config.num_hidden_layers

        if layer_idx >= n_layers:
            raise ValueError(
                f"layer_idx={layer_idx} but model only has {n_layers} layers"
            )

        # Create memory block (trainable, same dtype as base model)
        memory_block = MemoryBlock(
            d_model=d_model,
            memory_dim=memory_dim,
            n_heads=n_heads,
        ).to(device=device, dtype=dtype)

        print(f"Memory block: {memory_block.param_count_str()} trainable parameters")
        print(f"Spliced at layer {layer_idx}/{n_layers}")

        return cls(base_model, tokenizer, memory_block, layer_idx, device)

    def save_memory_block(self, path):
        """Save only the trained memory block weights."""
        torch.save(self.memory_block.state_dict(), path)
        print(f"Saved memory block to {path}")

    def load_memory_block(self, path):
        """Load trained memory block weights."""
        state_dict = torch.load(path, weights_only=True, map_location=self.device)
        self.memory_block.load_state_dict(state_dict)
        print(f"Loaded memory block from {path}")
