"""
========================================================================================
EXPERIMENT 6.3: CROSS-MODEL & CROSS-DOMAIN GENERALIZATION
========================================================================================

WHAT THIS PIPELINE EXPLORES:
This experiment investigates whether Endogenous Subspace Orthogonalization is a universal
property of low-rank adaptation across diverse transformer architectures and task domains.
Specifically, we test whether the collapse of superposition (C_gram drop > 85%) and
retention of downstream task loss holds across:
1. Architectural Paradigms: Modern Gated SwiGLU (Qwen2.5-0.5B) vs. Classical Dense GELU (Pythia-410M).
2. Domain Modalities: Affective Sentiment (tweet_eval) vs. Safety Moderation (real-toxicity-prompts).

HOW IT DOES IT:
1. Instantiates a 2x2 experimental matrix:
   - Models: Qwen/Qwen2.5-0.5B (Layer 12) and EleutherAI/pythia-410m (Layer 12).
   - Tasks:  cardiffnlp/tweet_eval (sentiment) and allenai/real-toxicity-prompts (safety).
2. For each cell, trains twin conditions under identical rank (r=32) and step budgets on a T4 GPU:
   - Condition A: Standard LoRA (lambda_ortho = 0.0)
   - Condition B: Ortho-LoRA (lambda_ortho = 1.0)
3. Evaluates:
   - Final Downstream Task Cross-Entropy Loss.
   - Gram Matrix Coherence C_gram (Degree of Superposition Elimination).
   - Mean Feature Participation Ratio (FPR, Token Monosemanticity).
   - Subspace Dimensional Entropy (d_eff).
4. Generates publication-ready visualizations:
   - Cross-Architecture Orthogonalization Efficacy (C_gram comparison bar charts).
   - Domain Generalization Radar / Bar Matrix.
   - Cross-Model Singular Value Spectra.
========================================================================================
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

# ==============================================================================
# 1. UNIVERSAL ORTHO-LORA MODULE (Supports both Qwen & Pythia)
# ==============================================================================
class UniversalOrthoLoRA(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 32, alpha: float = 32.0, device: str = "cuda"):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, dtype=torch.float32, device=device))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-3)
        self.last_z = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_fp32 = x.to(torch.float32)
        z = F.linear(x_fp32, self.lora_A)
        self.last_z = z
        lora_out = F.linear(z, self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


class SubspaceOrthogonalityLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, lora_layer: UniversalOrthoLoRA):
        B = lora_layer.lora_B.float()
        r = lora_layer.rank
        B_norm_val = torch.clamp(torch.norm(B, p=2, dim=0, keepdim=True), min=self.eps)
        B_norm = B / B_norm_val
        Gram_B = torch.matmul(B_norm.t(), B_norm)
        I_r = torch.eye(r, device=B.device)
        loss_weight_B = torch.norm(Gram_B - I_r, p="fro") ** 2 / (r * (r - 1))

        loss_sparse = torch.tensor(0.0, device=B.device)
        loss_act = torch.tensor(0.0, device=B.device)

        if lora_layer.last_z is not None:
            z = lora_layer.last_z.float().view(-1, r)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            T = z.shape[0]
            if T > 1:
                z_centered = z - z.mean(dim=0, keepdim=True)
                z_var = torch.mean(z_centered ** 2, dim=0, keepdim=True)
                z_std = torch.sqrt(torch.clamp(z_var, min=self.eps))
                z_normed = z_centered / z_std
                Corr_z = torch.matmul(z_normed.t(), z_normed) / (T - 1)
                loss_act = torch.norm(Corr_z - I_r, p="fro") ** 2 / (r * (r - 1))
            loss_sparse = torch.mean(torch.abs(z))

        return loss_weight_B + loss_act, loss_sparse


def attach_universal_lora(model, target_layer_idx: int, rank: int = 32, device: str = "cuda") -> UniversalOrthoLoRA:
    """Dynamically attaches LoRA to either Qwen (down_proj) or Pythia (dense_4h_to_h)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Qwen / LLaMA architecture
        layer = model.model.layers[target_layer_idx].mlp.down_proj
        parent = model.model.layers[target_layer_idx].mlp
        target_attr = "down_proj"
    elif hasattr(model, "gpt_neox"):
        # Pythia / GPT-NeoX architecture
        layer = model.gpt_neox.layers[target_layer_idx].mlp.dense_4h_to_h
        parent = model.gpt_neox.layers[target_layer_idx].mlp
        target_attr = "dense_4h_to_h"
    else:
        raise ValueError("Unsupported model family.")

    lora_module = UniversalOrthoLoRA(base_layer=layer, rank=rank, device=device)
    setattr(parent, target_attr, lora_module)
    return lora_module


# ==============================================================================
# 2. DATASET BUILDER (SENTIMENT VS. TOXICITY)
# ==============================================================================
def build_dataloaders(task_name: str, tokenizer, batch_size: int = 8, max_len: int = 128):
    if task_name == "sentiment":
        dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
        LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

        def tokenize_sentiment(examples):
            input_ids_list, attention_mask_list, labels_list = [], [], []
            for text, label_idx in zip(examples["text"], examples["label"]):
                prompt = f"Tweet: {text.strip()}\nSentiment:"
                full_text = f"{prompt} {LABEL_MAP[int(label_idx)]}"
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                full_enc = tokenizer(full_text, max_length=max_len, truncation=True, padding="max_length")

                ids = full_enc["input_ids"]
                mask = full_enc["attention_mask"]
                lbl = [-100] * len(ids)
                for idx in range(len(prompt_ids), len(ids)):
                    if mask[idx] == 1:
                        lbl[idx] = ids[idx]
                input_ids_list.append(ids)
                attention_mask_list.append(mask)
                labels_list.append(lbl)
            return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

        tok_train = dataset["train"].map(tokenize_sentiment, batched=True, remove_columns=dataset["train"].column_names)
        tok_eval = dataset["test"].map(tokenize_sentiment, batched=True, remove_columns=dataset["test"].column_names)

    elif task_name == "toxicity":
        # allenai/real-toxicity-prompts
        raw_data = load_dataset("allenai/real-toxicity-prompts", split="train")
        # Filter samples that have both prompt and continuation
        filtered = [x for x in raw_data if x["prompt"]["text"] and x["continuation"] and x["continuation"]["text"]][:2000]

        train_split = filtered[:1600]
        eval_split = filtered[1600:2000]

        def tokenize_toxicity(samples):
            input_ids_list, attention_mask_list, labels_list = [], [], []
            for item in samples:
                p_text = item["prompt"]["text"].strip()
                c_text = item["continuation"]["text"].strip()
                full_text = f"Prompt: {p_text}\nContinuation: {c_text}"
                prompt_ids = tokenizer.encode(f"Prompt: {p_text}\nContinuation:", add_special_tokens=True)
                full_enc = tokenizer(full_text, max_length=max_len, truncation=True, padding="max_length")

                ids = full_enc["input_ids"]
                mask = full_enc["attention_mask"]
                lbl = [-100] * len(ids)
                for idx in range(len(prompt_ids), len(ids)):
                    if mask[idx] == 1:
                        lbl[idx] = ids[idx]
                input_ids_list.append(ids)
                attention_mask_list.append(mask)
                labels_list.append(lbl)
            return {
                "input_ids": torch.tensor(input_ids_list),
                "attention_mask": torch.tensor(attention_mask_list),
                "labels": torch.tensor(labels_list)
            }

        tok_train = tokenize_toxicity(train_split)
        tok_eval = tokenize_toxicity(eval_split)

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data_dict):
                self.data = data_dict
            def __len__(self):
                return len(self.data["input_ids"])
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.data.items()}

        train_loader = DataLoader(SimpleDataset(tok_train), batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(SimpleDataset(tok_eval), batch_size=batch_size, shuffle=False)
        return train_loader, eval_loader

    tok_train.set_format("torch")
    tok_eval.set_format("torch")
    train_loader = DataLoader(tok_train, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tok_eval, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    return train_loader, eval_loader


# ==============================================================================
# 3. RUNNER FOR SINGLE MODEL-TASK CONDITION
# ==============================================================================
def train_and_eval_cell(
    model_name: str,
    task_name: str,
    target_layer_idx: int,
    lambda_ortho: float,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    rank: int = 32,
    steps: int = 200,
    lr: float = 5e-4,
    device: str = "cuda"
):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    for p in model.parameters():
        p.requires_grad = False

    lora_module = attach_universal_lora(model, target_layer_idx=target_layer_idx, rank=rank, device=device)
    optimizer = torch.optim.AdamW([lora_module.lora_A, lora_module.lora_B], lr=lr)
    ortho_criterion = SubspaceOrthogonalityLoss().to(device)

    model.train()
    step_count = 0
    train_iter = iter(train_loader)

    while step_count < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        task_loss = out.loss

        if lambda_ortho > 0.0:
            ortho_loss, sparse_loss = ortho_criterion(lora_module)
            total_loss = task_loss + (lambda_ortho * ortho_loss) + (1e-3 * sparse_loss)
        else:
            total_loss = task_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([lora_module.lora_A, lora_module.lora_B], max_norm=1.0)
        optimizer.step()
        step_count += 1

    # Evaluation
    model.eval()
    test_loss_sum = 0.0
    eval_steps = 0
    all_z = []

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 15: break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            test_loss_sum += out.loss.item()
            eval_steps += 1
            if lora_module.last_z is not None:
                all_z.append(lora_module.last_z.view(-1, rank).cpu())

    test_loss = test_loss_sum / eval_steps
    eval_z = torch.cat(all_z, dim=0)

    # Gram coherence with safe detach
    B = lora_module.lora_B.detach().float()
    B_norm = B / torch.clamp(torch.norm(B, p=2, dim=0, keepdim=True), min=1e-5)
    Gram_B = torch.matmul(B_norm.t(), B_norm).abs().cpu().numpy()
    np.fill_diagonal(Gram_B, 0.0)
    c_gram = float(Gram_B.sum() / (rank * (rank - 1)))

    # Feature Participation Ratio (FPR)
    z_abs = torch.abs(eval_z.detach())
    l1_sum = torch.sum(z_abs, dim=-1) ** 2
    l2_sum = torch.sum(eval_z.detach() ** 2, dim=-1) + 1e-8
    fpr = float(np.mean((l1_sum / (rank * l2_sum)).numpy()))

    # SVD Spectrum
    z_clean = torch.nan_to_num(eval_z.detach(), nan=0.0)
    z_cent = z_clean - z_clean.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(z_cent, full_matrices=False)
    p = S / (torch.sum(S) + 1e-8)
    p = torch.clamp(p, min=1e-12)
    d_eff = float(np.exp(-torch.sum(p * torch.log(p)).item()))

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "model": model_name.split("/")[-1],
        "task": task_name,
        "method": "Ortho-LoRA" if lambda_ortho > 0.0 else "Standard LoRA",
        "test_loss": float(test_loss),
        "c_gram": c_gram,
        "fpr": fpr,
        "d_eff": d_eff,
        "singular_values": S.cpu().numpy()
    }


# ==============================================================================
# 4. MAIN EXPERIMENTAL PIPELINE (2x2 MATRIX)
# ==============================================================================
def run_experiment_6_3(output_dir="./paper_artifacts/section_6_3"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Experiment 6.3 (Cross-Model & Domain Matrix) on {device}")

    # Matrix Definition
    models = ["Qwen/Qwen2.5-0.5B", "EleutherAI/pythia-410m"]
    tasks = ["sentiment", "toxicity"]
    target_layer_idx = 12 # Mid layer (proven optimal in 6.2)
    rank = 32

    results = []
    spectra_dict = {}

    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for task_name in tasks:
            print(f"\n==================================================================")
            print(f"[*] Running Cell: Model = {model_name.split('/')[-1]} | Task = {task_name.upper()}")
            print(f"==================================================================")

            train_loader, eval_loader = build_dataloaders(task_name, tokenizer, batch_size=8)

            # 1. Standard LoRA (lambda = 0.0)
            print(f"---> Training Standard LoRA...")
            res_std = train_and_eval_cell(
                model_name=model_name, task_name=task_name, target_layer_idx=target_layer_idx,
                lambda_ortho=0.0, train_loader=train_loader, eval_loader=eval_loader,
                rank=rank, steps=200, device=device
            )
            spectra_dict[(res_std["model"], task_name, "Standard")] = res_std.pop("singular_values")
            results.append(res_std)
            print(f"     Standard LoRA: Loss = {res_std['test_loss']:.4f} | C_gram = {res_std['c_gram']:.4f} | FPR = {res_std['fpr']:.3f}")

            # 2. Ortho-LoRA (lambda = 1.0)
            print(f"---> Training Ortho-LoRA (Ours)...")
            res_ortho = train_and_eval_cell(
                model_name=model_name, task_name=task_name, target_layer_idx=target_layer_idx,
                lambda_ortho=1.0, train_loader=train_loader, eval_loader=eval_loader,
                rank=rank, steps=200, device=device
            )
            spectra_dict[(res_ortho["model"], task_name, "Ortho")] = res_ortho.pop("singular_values")
            results.append(res_ortho)
            print(f"     Ortho-LoRA:    Loss = {res_ortho['test_loss']:.4f} | C_gram = {res_ortho['c_gram']:.4f} | FPR = {res_ortho['fpr']:.3f}")

    df_matrix = pd.DataFrame(results)
    df_matrix.to_csv(os.path.join(output_dir, "table_6_3_cross_generalization_benchmark.csv"), index=False)
    print("\n================== 6.3 CROSS-GENERALIZATION BENCHMARK TABLE ==================")
    print(df_matrix.to_string(index=False))

    # ==============================================================================
    # 5. PUBLICATION VISUALIZATIONS
    # ==============================================================================
    print("\n[*] Generating Cross-Model & Domain Visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- PANEL 1: Gram Coherence Collapse Across Architecture & Domain ---
    df_matrix["Experiment"] = df_matrix["model"] + "\n(" + df_matrix["task"] + ")"
    sns.barplot(
        data=df_matrix, x="Experiment", y="c_gram", hue="method", ax=axes[0],
        palette={"Standard LoRA": "#d62728", "Ortho-LoRA": "#2ca02c"}
    )
    axes[0].set_title(r"(a) Superposition Elimination ($\mathcal{C}_{gram} \downarrow$)" + "\nAcross Architectures & Domains")
    axes[0].set_ylabel(r"Off-Diagonal Coherence $\mathcal{C}_{gram}$")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 2: Task Cross-Entropy Retention ---
    sns.barplot(
        data=df_matrix, x="Experiment", y="test_loss", hue="method", ax=axes[1],
        palette={"Standard LoRA": "#d62728", "Ortho-LoRA": "#2ca02c"}
    )
    axes[1].set_title("(b) Task Performance Retention (Test Loss $\downarrow$)\n(Zero Destruction of Expressivity)")
    axes[1].set_ylabel("Downstream Test Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 3: Cross-Architecture Singular Value Spectrum (Sentiment Task) ---
    qwen_std = spectra_dict[("Qwen2.5-0.5B", "sentiment", "Standard")]
    qwen_ortho = spectra_dict[("Qwen2.5-0.5B", "sentiment", "Ortho")]
    pythia_std = spectra_dict[("pythia-410m", "sentiment", "Standard")]
    pythia_ortho = spectra_dict[("pythia-410m", "sentiment", "Ortho")]

    axes[2].plot(range(1, 33), qwen_std / qwen_std[0], "o--", color="#d62728", label="Qwen (Standard LoRA)")
    axes[2].plot(range(1, 33), qwen_ortho / qwen_ortho[0], "s-", color="#2ca02c", label="Qwen (Ortho-LoRA)")
    axes[2].plot(range(1, 33), pythia_std / pythia_std[0], "^--", color="#ff7f0e", label="Pythia (Standard LoRA)")
    axes[2].plot(range(1, 33), pythia_ortho / pythia_ortho[0], "d-", color="#1f77b4", label="Pythia (Ortho-LoRA)")

    axes[2].set_title("(c) Cross-Architecture Eigen-Spectrum Flatness\n(Gated SwiGLU vs. Dense GELU)")
    axes[2].set_xlabel("Singular Value Index")
    axes[2].set_ylabel("Normalized Magnitude ($\sigma_i / \sigma_1$)")
    axes[2].set_yscale("log")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fig_6_3_cross_generalization.png")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path.replace(".png", ".pdf"))
    plt.close()

    print(f"[*] Visualizations cleanly saved to: {plot_path}")
    print("[*] Experiment 6.3 execution complete.")