"""
Memory Block — the trainable neural module spliced into a frozen transformer.

Phase 1 (MVP): Read-only. Learns to retrieve and use persistent memories.
Phase 2: Write. Learns what information to store.

Architecture:
  READ:  hidden_state → query_proj → cross-attention to memory bank → output_proj → gated residual
  WRITE: hidden_state → importance_gate → key/value projections → store in memory bank
  GATE:  ReZero style (direct multiplication, init=0, NOT sigmoid)
"""

import torch
import torch.nn as nn


class MemoryBlock(nn.Module):
    def __init__(self, d_model, memory_dim=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.n_heads = n_heads

        assert memory_dim % n_heads == 0, (
            f"memory_dim ({memory_dim}) must be divisible by n_heads ({n_heads})"
        )

        # === READ PATHWAY ===
        self.read_query_proj = nn.Linear(d_model, memory_dim)
        self.read_attn = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.read_output_proj = nn.Linear(memory_dim, d_model)

        # === WRITE PATHWAY ===
        self.write_key_proj = nn.Linear(d_model, memory_dim)
        self.write_value_proj = nn.Linear(d_model, memory_dim)
        self.importance_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # === GATED RESIDUAL ===
        # Gate starts near zero — memory block contributes almost nothing at init.
        # Small non-zero value (0.01) so ALL params get gradient signal from step 1.
        # Pure ReZero (gate=0) only gives gate itself a gradient on the first step,
        # delaying learning for all other params until gate moves. 0.01 avoids this.
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.layer_norm = nn.LayerNorm(d_model)

        # NOTE: Do NOT zero-init read_output_proj. Combined with gate=0 (ReZero),
        # zero output proj creates a dead gradient (0 * 0 = 0, nothing learns).
        # Gate=0 alone is sufficient to start with zero memory contribution.
        # The non-zero output proj gives the gate a gradient signal to start learning.

    def read(self, hidden_state, memory_bank, device="cpu"):
        """
        Read from memory bank using cross-attention.

        hidden_state: [batch, seq_len, d_model]
        memory_bank: MemoryBank instance
        device: device for memory tensors

        Returns: modified hidden_state (same shape)
        """
        if memory_bank is None or not memory_bank.has_entries():
            return hidden_state

        batch_size = hidden_state.size(0)

        # Project to query space
        query = self.read_query_proj(hidden_state)  # [batch, seq_len, memory_dim]

        # Get memory contents and expand for batch
        mem_keys = memory_bank.get_keys(device=device)      # [n_mem, memory_dim]
        mem_values = memory_bank.get_values(device=device)   # [n_mem, memory_dim]

        mem_keys = mem_keys.unsqueeze(0).expand(batch_size, -1, -1)     # [batch, n_mem, memory_dim]
        mem_values = mem_values.unsqueeze(0).expand(batch_size, -1, -1) # [batch, n_mem, memory_dim]

        # Cross-attention: current tokens attend to stored memories
        retrieved, _ = self.read_attn(
            query=query,
            key=mem_keys,
            value=mem_values,
        )  # [batch, seq_len, memory_dim]

        # Project back to model space
        memory_output = self.read_output_proj(retrieved)  # [batch, seq_len, d_model]

        # Gated residual — gate starts at 0 (ReZero)
        hidden_state = hidden_state + self.gate * self.layer_norm(memory_output)

        return hidden_state

    def encode_to_memory(self, hidden_state):
        """
        Project hidden states into memory space for storage.

        hidden_state: [batch, seq_len, d_model]
        Returns: keys [batch, memory_dim], values [batch, memory_dim]
        """
        # Last token captures full sentence context in autoregressive models.
        # Mean-pool dilutes discriminative tokens ("Alice" in "The user's name is Alice")
        # because shared prefix tokens dominate the average.
        last_token = hidden_state[:, -1, :]  # [batch, d_model]

        keys = self.write_key_proj(last_token)    # [batch, memory_dim]
        values = self.write_value_proj(last_token) # [batch, memory_dim]

        return keys, values

    def forward(self, hidden_state, memory_bank, device="cpu"):
        """Full forward: read from memory."""
        return self.read(hidden_state, memory_bank, device=device)

    def param_count(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count_str(self):
        count = self.param_count()
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        return str(count)
