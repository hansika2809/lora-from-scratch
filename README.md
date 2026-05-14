# LoRA: Low-Rank Adaptation From-Scratch PyTorch Implementation

A clean, from-scratch reproduction of the LoRA paper (Hu et al., ICLR 2022) in PyTorch. No HuggingFace, no PEFT wrappers — just raw PyTorch to understand every moving part.

I built this to really internalize how parameter-efficient fine-tuning works under the hood. The official paper is elegant but dense, and I found that implementing it myself was the fastest way to go from "I get the math" to "I can actually build and debug this."

## What's LoRA?

The core problem: fine-tuning large language models means updating millions (or billions) of parameters. LoRA's insight is that the weight updates during fine-tuning have low intrinsic rank — you don't need a full-rank update to adapt a pretrained model.

Instead of updating the full weight matrix W, we learn a low-rank decomposition:


W' = W + (α/r) · B·A

where:
- W is the frozen pretrained weight (no gradients)
- A ∈ ℝ^(r × d_in) and **B** ∈ ℝ^(d_out × r) are small trainable matrices
- r << min(d_in, d_out) is the rank (typically 4-16)
- α is a scaling factor that lets you change r without retuning the learning rate

The beautiful part: B is initialized to zero, so ΔW = 0 at the start of training. The model starts identical to the pretrained one, and the low-rank adaptation is learned from there.

## Architecture


                    ┌─────────────────┐
    Input x ──────>│  Frozen W₀       │──────> Base output
         │         │  (no gradients)  │           │
         │         └─────────────────┘           │
         │                                        │  (+)  ──> Final output
         │         ┌──────┐  ┌──────┐            │
         └────────>│  A   │─>│  B   │──> ×(α/r) ─┘
                   │(r×d) │  │(d×r) │
                   └──────┘  └──────┘
                   Trainable LoRA path


Applied to a Transformer's attention layers:


┌──────────────────────────────────────────────┐
│              Transformer Block               │
│                                              │
│   ┌──────────────────────────────────────┐   │
│   │       Multi-Head Self-Attention      │   │
│   │                                      │   │
│   │   Q = LoRA(q_proj)(x)   ← adapted   │   │
│   │   K = k_proj(x)         ← frozen    │   │
│   │   V = LoRA(v_proj)(x)   ← adapted   │   │
│   │   Out = out_proj(attn)  ← frozen    │   │
│   │                                      │   │
│   └──────────────────────────────────────┘   │
│                    ↓                         │
│   ┌──────────────────────────────────────┐   │
│   │        Feed-Forward Network          │   │
│   │        (frozen, no LoRA)             │   │
│   └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘


## Project Structure


lora-from-scratch/
├── src/
│   ├── transformer.py    # GPT-2 style model from scratch
│   ├── lora.py           # LoRA layer, apply/merge/count utilities
│   ├── config.py         # Dataclass configs for model, LoRA, training
│   ├── dataset.py        # Shakespeare dataset + char tokenizer
│   └── utils.py          # Training loop, evaluation, plotting
├── train.py              # Full experiment (pretrain → FT → LoRA → compare)
├── demo.py               # Quick 2-min demo
├── requirements.txt      # torch, numpy, matplotlib 
└── README.md


## Quick Start

bash
# setup
pip install -r requirements.txt

# quick demo (~2-3 min on CPU)
python demo.py

# full experiment (~15-20 min on CPU, faster on GPU)
python train.py

# or run the full experiment in demo mode
python train.py --demo


## Key Implementation Details

### What I implemented faithfully from the paper:

1. Initialization strategy — A with Kaiming uniform, B with zeros. This is critical: it means ΔW = 0 at init, so the model behavior is preserved before training begins.

2. α/r scaling— The output of the LoRA path is scaled by α/r. This is a practical detail from Section 4.1 that lets you change rank without retuning the learning rate.

3. Target modules — By default, LoRA is applied to Q and V projections in attention (Section 4.2). The code supports targeting any linear layer.

4. Weight merging — After training, B·A can be merged directly into W for zero additional inference latency. This is verified in the experiment (pre/post merge loss should be identical).

### Design decisions I made:

- Separate Q/K/V projections instead of fused QKV — makes it cleaner to target individual projections with LoRA
- Pre-LayerNorm architecture (GPT-2 style) — more stable training
- Weight tying between input embeddings and LM head — standard practice, reduces parameters
- Character-level tokenization — keeps the implementation self-contained, no external tokenizer dependency

## Results

Running 'python train.py --demo` on a small GPT (128d, 4 heads, 4 layers):

| Method | Trainable Params | Val Loss | Training Time |
|--------|:---:|:---:|:---:|
| Pretrained (no FT) | 621K (100%) | ~1.8 | ~60s |
| Full Fine-tuning | 621K (100%) | ~1.5 | ~25s |
| LoRA (r=4, α=8) | ~4K (0.7%) | ~1.6 | ~25s |

Key takeaway: LoRA achieves ~95% of full fine-tuning performance with <1% of the trainable parameters.

Weight merging verification: pre-merge and post-merge losses match to <1e-6, confirming zero inference overhead.

*(Results will vary slightly due to random initialization. Run `train.py` for the full-scale experiment.)*

## What I Learned

A few things that clicked for me while implementing this:

1. **The zero-init of B is more than a trick** — it's what makes LoRA stable. Without it, you'd be adding random noise to a pretrained model's forward pass, which would be catastrophic for large models.

2. **The α/r scaling is underrated** — it's easy to skip this as a minor detail, but it's what makes LoRA practically usable. Without it, changing the rank would require a full hyperparameter search for learning rate.

3. **Module replacement in PyTorch is fiddly** — you can't just swap modules during `named_modules()` iteration. The approach of iterating parents and replacing children with `setattr` is the clean way.

4. **Weight tying + LoRA needs care** — since the embedding and LM head share weights, you shouldn't apply LoRA to both. I only target attention projections, which sidesteps this.

## References

```
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan
          and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

**Related reading:**
- [The original paper (arXiv)](https://arxiv.org/abs/2106.09685)
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for the transformer implementation
- [Sebastian Raschka's LoRA explainer](https://sebastianraschka.com/blog/2023/llm-finetuning.html) — great supplementary material

---

Built as a paper reproduction exercise. Feedback welcome.
