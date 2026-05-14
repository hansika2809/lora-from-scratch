

import os
import sys
import copy
import argparse
import torch
from torch.utils.data import DataLoader

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ModelConfig, TrainConfig, DemoModelConfig, DemoTrainConfig, LoRAConfig
from src.transformer import GPT
from src.lora import apply_lora, merge_lora, count_parameters, print_lora_summary, lora_state_dict
from src.dataset import prepare_data
from src.utils import (
    get_device, train_model, evaluate, generate_text,
    plot_comparison, print_comparison_table
)


def run_experiment(demo_mode=False):
    print("\n" + "=" * 55)
    print("  LoRA: Low-Rank Adaptation — Reproduction Experiment")
    print("  Paper: Hu et al., ICLR 2022")
    print("=" * 55)

    # ---- configs ----
    if demo_mode:
        print("\n  [Running in demo mode — smaller model, fewer epochs]")
        model_cfg = DemoModelConfig()
        pretrain_cfg = DemoTrainConfig()
        ft_cfg = DemoTrainConfig(epochs=3, lr=1e-4)
        lora_cfg = LoRAConfig(rank=4, alpha=8)
    else:
        model_cfg = ModelConfig()
        pretrain_cfg = TrainConfig(epochs=10, lr=3e-4)
        ft_cfg = TrainConfig(epochs=5, lr=1e-4)
        lora_cfg = LoRAConfig(rank=8, alpha=16)

    device = get_device()
    print(f"  Device: {device}\n")

    # ---- data ----
    tokenizer, train_set, val_set = prepare_data(
        context_len=model_cfg.context_len,
        data_dir="data"
    )
    model_cfg.vocab_size = tokenizer.vocab_size

    train_loader = DataLoader(
        train_set, batch_size=pretrain_cfg.batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=pretrain_cfg.batch_size,
        shuffle=False, num_workers=0, drop_last=True
    )

    # ================================================================
    # PHASE 1: Pretrain base model
    # ================================================================
    print("\n" + "=" * 55)
    print("  PHASE 1: Pretraining base GPT model")
    print("=" * 55)

    base_model = GPT(model_cfg)
    base_counts = count_parameters(base_model)
    print(f"  Model parameters: {base_counts['total']:,}")

    pretrain_history = train_model(
        base_model, train_loader, val_loader,
        pretrain_cfg, device, label="Pretrain"
    )

    # save pretrained checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    pretrained_state = copy.deepcopy(base_model.state_dict())
    if pretrain_cfg.save_checkpoint:
        torch.save(pretrained_state, "checkpoints/pretrained.pt")
        print("  Saved pretrained checkpoint.\n")

    # generate a sample from pretrained model
    print("  --- Sample from pretrained model ---")
    sample = generate_text(base_model, tokenizer, device, prompt="ROMEO:")
    print(f"  {sample[:300]}")
    print()

    # ================================================================
    # PHASE 2: Full fine-tuning (baseline)
    # ================================================================
    print("=" * 55)
    print("  PHASE 2: Full fine-tuning (all parameters)")
    print("=" * 55)

    ft_model = GPT(model_cfg)
    ft_model.load_state_dict(copy.deepcopy(pretrained_state))

    ft_counts = count_parameters(ft_model)
    print(f"  Trainable parameters: {ft_counts['trainable']:,} (100%)")

    ft_loader = DataLoader(
        train_set, batch_size=ft_cfg.batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )
    ft_history = train_model(
        ft_model, ft_loader, val_loader,
        ft_cfg, device, label="Full Fine-tune"
    )

    # ================================================================
    # PHASE 3: LoRA fine-tuning
    # ================================================================
    print("=" * 55)
    print("  PHASE 3: LoRA fine-tuning (low-rank adaptation)")
    print("=" * 55)

    lora_model = GPT(model_cfg)
    lora_model.load_state_dict(copy.deepcopy(pretrained_state))

    # apply LoRA to the attention projections
    replaced = apply_lora(
        lora_model,
        target_modules=lora_cfg.target_modules,
        rank=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
    )
    print_lora_summary(lora_model, replaced)

    lora_counts = count_parameters(lora_model)

    lora_loader = DataLoader(
        train_set, batch_size=ft_cfg.batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )
    lora_history = train_model(
        lora_model, lora_loader, val_loader,
        ft_cfg, device, label="LoRA Fine-tune"
    )

    # save just the LoRA weights — they're tiny
    lora_weights = lora_state_dict(lora_model)
    if ft_cfg.save_checkpoint:
        torch.save(lora_weights, "checkpoints/lora_weights.pt")
        lora_size = os.path.getsize("checkpoints/lora_weights.pt")
        base_size = os.path.getsize("checkpoints/pretrained.pt")
        print(f"  LoRA weights: {lora_size / 1024:.1f} KB "
              f"(vs base model: {base_size / 1024:.1f} KB)")

    # ================================================================
    # PHASE 4: Comparison
    # ================================================================
    print("\n" + "=" * 55)
    print("  PHASE 4: Results comparison")
    print("=" * 55)

    # evaluate both fine-tuned models
    ft_val_loss = evaluate(ft_model, val_loader, device)
    lora_val_loss = evaluate(lora_model, val_loader, device)
    pretrain_val_loss = evaluate(base_model, val_loader, device)

    results = [
        {
            "method": "Pretrained (no FT)",
            "params": base_counts["trainable"],
            "train_loss": pretrain_history["train_loss"][-1],
            "val_loss": pretrain_val_loss,
        },
        {
            "method": "Full Fine-tuning",
            "params": ft_counts["trainable"],
            "train_loss": ft_history["train_loss"][-1],
            "val_loss": ft_val_loss,
        },
        {
            "method": f"LoRA (r={lora_cfg.rank}, α={lora_cfg.alpha})",
            "params": lora_counts["trainable"],
            "train_loss": lora_history["train_loss"][-1],
            "val_loss": lora_val_loss,
        },
    ]
    print_comparison_table(results)

    # parameter efficiency
    reduction = (1 - lora_counts["trainable"] / ft_counts["trainable"]) * 100
    print(f"  LoRA reduces trainable params by {reduction:.1f}%")
    print(f"  ({lora_counts['trainable']:,} vs {ft_counts['trainable']:,})\n")

    # plot
    plot_comparison(
        [ft_history, lora_history],
        ["Full Fine-tuning", f"LoRA (r={lora_cfg.rank})"],
        save_path="figures/comparison.png"
    )

    # test weight merging
    print("  --- Testing weight merging (zero-latency inference) ---")
    pre_merge_loss = evaluate(lora_model, val_loader, device)
    merge_lora(lora_model)
    post_merge_loss = evaluate(lora_model, val_loader, device)
    print(f"  Pre-merge val loss:  {pre_merge_loss:.6f}")
    print(f"  Post-merge val loss: {post_merge_loss:.6f}")
    print(f"  Difference: {abs(pre_merge_loss - post_merge_loss):.8f} "
          f"(should be ~0)")

    # generate samples
    print("\n  --- Generated text comparison ---")
    print("\n  [Full Fine-tuning]:")
    sample_ft = generate_text(ft_model, tokenizer, device, prompt="JULIET:")
    print(f"  {sample_ft[:300]}")
    print("\n  [LoRA Fine-tuning]:")
    sample_lora = generate_text(lora_model, tokenizer, device, prompt="JULIET:")
    print(f"  {sample_lora[:300]}")

    print("\n" + "=" * 55)
    print("  Experiment complete!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA paper reproduction experiment")
    parser.add_argument("--demo", action="store_true",
                        help="Run with smaller model for quick testing")
    args = parser.parse_args()
    run_experiment(demo_mode=args.demo)
