"""
demo.py — Quick demo of LoRA vs full fine-tuning.


This uses a tiny model and a data subset so it's fast, but still
demonstrates the core concepts: parameter efficiency, weight merging,
and comparable performance.
"""

import os
import sys
import copy
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import LoRAConfig
from src.transformer import GPT
from src.lora import (
    LoRALinear, apply_lora, merge_lora, unmerge_lora,
    count_parameters, print_lora_summary
)
from src.dataset import prepare_data
from src.utils import get_device, train_model, evaluate, generate_text


def main():
    print()
    print("=" * 55)
    print("  LoRA from Scratch -- Quick Demo")
    print("  Paper: Hu et al., ICLR 2022")
    print("=" * 55)

    device = get_device()
    print(f"\n  Device: {device}")

    # ---- tiny model config for fast demo ----
    from dataclasses import dataclass

    @dataclass
    class TinyConfig:
        d_model: int = 64
        n_heads: int = 4
        n_layers: int = 3
        context_len: int = 32
        vocab_size: int = 65
        dropout: float = 0.1
        bias: bool = False

    @dataclass
    class TinyTrainConfig:
        batch_size: int = 64
        lr: float = 1e-3
        weight_decay: float = 0.01
        epochs: int = 3
        warmup_steps: int = 0
        grad_clip: float = 1.0
        eval_interval: int = 1
        save_checkpoint: bool = False

    # ---- data ----
    print("\n  Loading data...")
    tokenizer, train_set, val_set = prepare_data(
        context_len=32, data_dir="data"
    )

    # use a subset for the demo — full dataset is too slow on CPU
    max_train = 20000
    max_val = 2000
    if len(train_set) > max_train:
        train_set = Subset(train_set, range(max_train))
    if len(val_set) > max_val:
        val_set = Subset(val_set, range(max_val))
    print(f"  Using subset: {len(train_set)} train, {len(val_set)} val samples")

    model_cfg = TinyConfig(vocab_size=tokenizer.vocab_size)
    train_cfg = TinyTrainConfig()

    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=64, shuffle=False, drop_last=True
    )

    # ---- pretrain ----
    print("\n  Step 1: Quick pretrain...")
    model = GPT(model_cfg)
    base_params = count_parameters(model)
    pretrain_history = train_model(
        model, train_loader, val_loader,
        train_cfg, device, label="Pretrain"
    )
    pretrained_state = copy.deepcopy(model.state_dict())

    # ---- full fine-tuning ----
    print("  Step 2: Full fine-tuning...")
    ft_model = GPT(model_cfg)
    ft_model.load_state_dict(copy.deepcopy(pretrained_state))
    ft_cfg = TinyTrainConfig(epochs=2, lr=5e-4)
    ft_history = train_model(
        ft_model, train_loader, val_loader,
        ft_cfg, device, label="Full FT"
    )

    # ---- LoRA fine-tuning ----
    print("  Step 3: LoRA fine-tuning...")
    lora_model = GPT(model_cfg)
    lora_model.load_state_dict(copy.deepcopy(pretrained_state))

    replaced = apply_lora(lora_model, target_modules=["q_proj", "v_proj"],
                          rank=4, alpha=8)
    print_lora_summary(lora_model, replaced)

    lora_params = count_parameters(lora_model)
    lora_history = train_model(
        lora_model, train_loader, val_loader,
        ft_cfg, device, label="LoRA FT"
    )

    # ---- results ----
    ft_val = evaluate(ft_model, val_loader, device)
    lora_val = evaluate(lora_model, val_loader, device)

    ft_trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
    lora_trainable = lora_params["trainable"]

    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)
    print(f"  Full FT:  {ft_trainable:>8,} params | val loss: {ft_val:.4f}")
    print(f"  LoRA FT:  {lora_trainable:>8,} params | val loss: {lora_val:.4f}")
    reduction = (1 - lora_trainable / ft_trainable) * 100
    print(f"  Parameter reduction: {reduction:.1f}%")
    print("=" * 55)

    # ---- verify weight merging ----
    print("\n  Verifying weight merging (LoRA's zero-latency trick)...")
    pre = evaluate(lora_model, val_loader, device)
    merge_lora(lora_model)
    post = evaluate(lora_model, val_loader, device)
    print(f"    Before merge: {pre:.6f}")
    print(f"    After merge:  {post:.6f}")
    diff = abs(pre - post)
    match_str = "YES" if diff < 1e-5 else "NO"
    print(f"    Match: {match_str} (diff={diff:.8f})")

    # ---- generate ----
    print("\n  Generated text sample (LoRA model):")
    print("  " + "-" * 50)
    sample = generate_text(lora_model, tokenizer, device,
                           prompt="ROMEO:", max_tokens=150)
    print(f"  {sample[:250]}")
    print("  " + "-" * 50)

    print("\n  Done! For the full experiment, run: python train.py\n")


if __name__ == "__main__":
    main()
