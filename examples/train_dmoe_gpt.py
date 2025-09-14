import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# Ensure repo root is on sys.path when running as a script
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from megablocks.layers import dMoE
from megablocks.layers.arguments import Arguments
from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss
from megablocks.layers.router import batched_router_zloss, clear_router_zloss


class BinTokenDataset(torch.utils.data.Dataset):
    """Simple token dataset backed by a .bin file of integer token ids.

    Expected dtype is int32 (default) or uint16. Returns contiguous blocks
    of length seq_len+1 so the next token is the target.
    """

    def __init__(self, path: str, seq_len: int, dtype: str = "uint16", start_item: int = 0, num_items: Optional[int] = None):
        super().__init__()
        if dtype not in ("int32", "uint16"):
            raise ValueError("dtype must be 'int32' or 'uint16'")
        np_dtype = np.int32 if dtype == "int32" else np.uint16
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        self._arr = np.memmap(path, dtype=np_dtype, mode="r")
        self.seq_len = seq_len

        # Number of full sequences available (leave 1 for next-token target)
        self._num_tokens = len(self._arr)
        if self._num_tokens <= seq_len:
            raise ValueError(
                f"Dataset too small: {self._num_tokens} tokens, need > {seq_len}"
            )
        total_items = (self._num_tokens - 1) // seq_len
        # Subrange selection
        if start_item < 0:
            start_item = 0
        if start_item > total_items:
            start_item = total_items
        if num_items is None:
            num_items = max(0, total_items - start_item)
        else:
            num_items = max(0, min(num_items, total_items - start_item))
        self._start_item = start_item
        self._num_items = num_items

    def __len__(self) -> int:
        return self._num_items

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        global_idx = self._start_item + idx
        start = global_idx * self.seq_len
        end = start + self.seq_len + 1
        x = torch.from_numpy(np.array(self._arr[start:end - 1], copy=False)).long()
        y = torch.from_numpy(np.array(self._arr[start + 1:end], copy=False)).long()
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False,
        )
        # Precompute causal mask (seq, seq)
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        attn_mask = self.attn_mask[:seq_len, :seq_len]
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return out


class MoEFeedForward(nn.Module):
    """Wraps MegaBlocks dMoE as the FFN in a Transformer block."""

    def __init__(self, args: Arguments):
        super().__init__()
        # Return a single tensor (bias added inside) to simplify residuals.
        args = Arguments(**{**args.__dict__, "return_bias": False})
        self.moe = dMoE(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        dmoe_args: Arguments,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(n_heads, hidden_size, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ff = MoEFeedForward(dmoe_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq, batch, hidden]
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLMHeadModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_layers: int,
        hidden_size: int,
        n_heads: int,
        dmoe_args: Arguments,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        blocks = []
        # Ensure dMoE args knows number of layers and hidden size
        dmoe_args = Arguments(**{
            **dmoe_args.__dict__,
            "num_layers": n_layers,
            "hidden_size": hidden_size,
        })
        for _ in range(n_layers):
            blocks.append(
                TransformerBlock(hidden_size, n_heads, dropout, max_seq_len, dmoe_args)
            )
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [batch, seq]
        bsz, seq_len = idx.size()
        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)  # [1, seq]
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = x.transpose(0, 1)  # [seq, batch, hidden]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.transpose(0, 1)  # [batch, seq, hidden]
        logits = self.lm_head(x)
        return logits


@dataclass
class TrainConfig:
    # Data
    train_bin: str = "train.bin"
    val_bin: Optional[str] = None
    bin_dtype: str = "uint16"  # or "int32"
    seq_len: int = 1024
    vocab_size: int = 32768

    # Model
    n_layers: int = 12
    n_heads: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 1536  # typical 2x hidden for small dMoE
    num_experts: int = 8
    top_k: int = 1
    moe_loss_weight: float = 0.1
    moe_zloss_weight: float = 0.0
    mlp_impl: str = "grouped"  # "grouped" or "sparse"
    dropout: float = 0.0

    # Optim
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 20000
    batch_size: int = 8
    num_workers: int = 2

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Misc
    device: str = "cuda"
    log_interval: int = 100
    eval_interval: int = 1000
    ckpt_dir: Optional[str] = None


def cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * t))


def create_dmoe_args(cfg: TrainConfig, device: torch.device) -> Arguments:
    return Arguments(
        hidden_size=cfg.hidden_size,
        ffn_hidden_size=cfg.ffn_hidden_size,
        num_layers=cfg.n_layers,
        moe_num_experts=cfg.num_experts,
        moe_top_k=cfg.top_k,
        moe_loss_weight=cfg.moe_loss_weight,
        moe_zloss_weight=cfg.moe_zloss_weight,
        moe_expert_model_parallelism=False,
        pipeline_model_parallel_size=1,
        memory_optimized_mlp=True,
        mlp_type="mlp",
        mlp_impl=cfg.mlp_impl,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        device=device,
        return_bias=False,
    )


def train(cfg: TrainConfig):
    assert torch.cuda.is_available(), "CUDA is required for dMoE kernels"
    # Normalize device string: allow "cuda" -> "cuda:0"
    device_str = cfg.device
    if device_str.startswith("cuda") and (":" not in device_str or device_str.endswith(":")):
        device_str = "cuda:0"
    device = torch.device(device_str)
    # set_device expects an index or a device with an index
    torch.cuda.set_device(device.index if device.index is not None else 0)

    # Data: if no val.bin provided, split 1% from train.bin for validation.
    base_ds = BinTokenDataset(cfg.train_bin, cfg.seq_len, dtype=cfg.bin_dtype)
    if cfg.val_bin:
        train_ds = base_ds
        val_ds = BinTokenDataset(cfg.val_bin, cfg.seq_len, dtype=cfg.bin_dtype)
    else:
        total_items = len(base_ds)
        val_items = max(1, int(math.ceil(0.01 * total_items)))
        train_items = total_items - val_items
        if train_items <= 0:
            raise ValueError("Dataset too small to create 1% validation split.")
        train_ds = BinTokenDataset(cfg.train_bin, cfg.seq_len, dtype=cfg.bin_dtype, start_item=0, num_items=train_items)
        val_ds = BinTokenDataset(cfg.train_bin, cfg.seq_len, dtype=cfg.bin_dtype, start_item=train_items, num_items=val_items)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
        )

    # Model
    dmoe_args = create_dmoe_args(cfg, device)
    model = GPTLMHeadModel(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.seq_len,
        n_layers=cfg.n_layers,
        hidden_size=cfg.hidden_size,
        n_heads=cfg.n_heads,
        dmoe_args=dmoe_args,
        dropout=cfg.dropout,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # Training loop
    global_step = 0
    model.train()
    while global_step < cfg.max_steps:
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # LR schedule per step
            lr_now = cosine_lr(global_step, cfg.max_steps, cfg.lr, cfg.min_lr, cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            autocast_dtype = torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else None)
            ctx = torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype)
            with ctx:
                logits = model(x)
                # Flatten for CE: [(batch*seq), vocab]
                loss_lm = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
                )
                # Add dMoE auxiliary losses (already scaled by args weights)
                loss_lbl = batched_load_balancing_loss(dmoe_args)
                loss_z = batched_router_zloss(dmoe_args) if cfg.moe_zloss_weight > 0 else 0.0
                loss = loss_lm + loss_lbl + (loss_z if isinstance(loss_z, torch.Tensor) else 0.0)

            if cfg.fp16:
                scaler.scale(loss).backward()
                if cfg.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            # Clear accumulators for next batch
            clear_load_balancing_loss()
            clear_router_zloss()

            if (global_step + 1) % cfg.log_interval == 0:
                ppl = math.exp(loss_lm.item()) if loss_lm.item() < 20 else float("inf")
                print(
                    f"step {global_step+1} | lr {lr_now:.3e} | loss {loss.item():.4f} | lm {loss_lm.item():.4f} | ppl {ppl:.2f}"
                )

            if cfg.eval_interval and (global_step + 1) % cfg.eval_interval == 0 and val_loader is not None:
                evaluate(model, val_loader, dmoe_args, cfg, device)

            if cfg.ckpt_dir and (global_step + 1) % cfg.eval_interval == 0:
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step + 1,
                }, os.path.join(cfg.ckpt_dir, f"ckpt_{global_step+1}.pt"))

            global_step += 1
            if global_step >= cfg.max_steps:
                break


@torch.no_grad()
def evaluate(model: nn.Module, loader, dmoe_args: Arguments, cfg: TrainConfig, device: torch.device):
    model.eval()
    total_loss = 0.0
    n = 0
    autocast_dtype = torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else None)
    ctx = torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        clear_load_balancing_loss()
        clear_router_zloss()
    avg = total_loss / max(1, n)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    print(f"eval | loss {avg:.4f} | ppl {ppl:.2f}")
    model.train()


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a small GPT with MegaBlocks dMoE (no Megatron)")
    # Data
    p.add_argument("--train-bin", type=str, required=True)
    p.add_argument("--val-bin", type=str, default=None)
    p.add_argument("--bin-dtype", type=str, default="uint16", choices=["int32", "uint16"])
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=32768)
    # Model
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--ffn-hidden-size", type=int, default=1536)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--mlp-impl", type=str, default="grouped", choices=["grouped", "sparse"])
    p.add_argument("--moe-loss-weight", type=float, default=0.1)
    p.add_argument("--moe-zloss-weight", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    # Optim
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    # Precision
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    # Misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--ckpt-dir", type=str, default=None)

    a = p.parse_args()
    return TrainConfig(
        train_bin=a.train_bin,
        val_bin=a.val_bin,
        bin_dtype=a.bin_dtype,
        seq_len=a.seq_len,
        vocab_size=a.vocab_size,
        n_layers=a.n_layers,
        n_heads=a.n_heads,
        hidden_size=a.hidden_size,
        ffn_hidden_size=a.ffn_hidden_size,
        num_experts=a.num_experts,
        top_k=a.top_k,
        moe_loss_weight=a.moe_loss_weight,
        moe_zloss_weight=a.moe_zloss_weight,
        mlp_impl=a.mlp_impl,
        dropout=a.dropout,
        lr=a.lr,
        min_lr=a.min_lr,
        weight_decay=a.weight_decay,
        betas=(a.beta1, a.beta2),
        grad_clip=a.grad_clip,
        warmup_steps=a.warmup_steps,
        max_steps=a.max_steps,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        bf16=a.bf16 and not a.fp16,
        fp16=a.fp16,
        device=a.device,
        log_interval=a.log_interval,
        eval_interval=a.eval_interval,
        ckpt_dir=a.ckpt_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
