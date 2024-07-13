# Temporary home for a GPT implementation; expect to move this up to llm_playbooks later
import math

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.nn import functional as F

# batch_size = 64 # how many independent sequences will we process
# block_size = 256 # what is the maximum context length for predictions
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2


class GPTConfig(BaseModel):
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


GPT2_CONFIG = GPTConfig(
    block_size=1024,  # max sequence lengths
    vocab_size=50257,  # 50k BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer=12,
    n_head=12,
    n_embd=768,
)


class CausalSelfAttention(nn.Module):
    # One way to think about self attention is as a pooling layer

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # concatenated KQV projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # ?
        blksz = config.block_size
        self.register_buffer(
            "bias", torch.tril(torch.ones(blksz, blksz)).view(1, 1, blksz, blksz)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, n_embd
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # each of these end up being of size (B, nh, T, hs), where
        # nh: number of heads; hs: head size; C: # of channels = nh * hs
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, T, T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)  # (B, T, C) x (C, C) -> (B, T, C)

        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")  # tanh only for historical purposes
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} should not exceed block size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer. ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
