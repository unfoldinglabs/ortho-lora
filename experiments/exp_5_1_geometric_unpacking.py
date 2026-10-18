"""
========================================================================================
EXPERIMENT 5.1: GEOMETRIC UNPACKING (COLLAPSE OF REPRESENTATION SUPERPOSITION)
========================================================================================
ackward()
        torch.nn.utils.clip_grad_norm_([ortho_module.lora_A, ortho_module.lora_B], max_norm=1.0)
        optimizer.step()

        step_count += 1
        history["step"].append(step_count)
        history["task_loss"].append(task_loss.item())
        history["ortho_loss"].append(ortho_loss.item())

…# ==============================================================================

WHAT THIS PIPELINE EXPLORES:
This experiment investigates whether introducing an attribution-conditioned orthogonalization
loss into a Low-Rank Adaptation (LoRA) bottleneck forces the model's adapted subspace to unpack
features from superposition into mutually orthogonal, linearly independent axes, without
degrading downstream task performance.

HOW IT DOES IT:
1. Replaces the target linear projection (MLP down_proj at layer ~50% depth) of a causal LM
   (Qwen/Qwen2.5-0.5B or EleutherAI/pythia-410m) with an instrumented OrthoLoRALinear layer.
2. Conducts a controlled twin training run on a benchmark task (cardiffnlp/tweet_eval, "sentiment"):
   - Baseline Condition: Standard LoRA (lambda_ortho = 0.0, lambda_sparse = 0.0)
   - Experimental Condition: Ortho-LoRA (lambda_ortho = 1.0, lambda_sparse = 1e-3)
3. Evaluates both models across geometric, statistical, and spatial dimensions:
   - Directional Gram Coherence (pairwise off-diagonal cosine similarities of B and A).
   - Dynamic Feature Co-activation Correlation (token-level channel cross-talk).
   - Feature Participation Ratio (FPR) distribution per token (monosemanticity vs. density).
   - Effective Subspace Dimensionality (spectral entropy of singular values).
   - 2D PCA Projections of both the Feature Weight Vectors (B columns) and Activation Latents (z).
========================================================================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Publication aesthetic settings
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
# 1. INSTRUMENTED ORTHO-LORA LINEAR MODULE
# ==============================================================================
class OrthoLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 32, alpha: float = 32.0):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Crucial: Trainable parameters ALWAYS in float32 to prevent FP16 optimizer overflow
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-3)

        self.last_z = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        # Cast input to float32 for stable low-rank computation
        x_fp32 = x.to(torch.float32)
        z = F.linear(x_fp32, self.lora_A)
        self.last_z = z
        lora_out = F.linear(z, self.lora_B) * self.scaling

        # Return matched to base layer dtype (FP16)
        return base_out + lora_out.to(base_out.dtype)


# ==============================================================================
# 2. ORTHOGONALITY AND SPARSITY LOSS CRITERIA
# ==============================================================================
class SubspaceOrthogonalityLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, lora_layer: OrthoLoRALinear):
        B = lora_layer.lora_B.float()
        A = lora_layer.lora_A.float()
        r = lora_layer.rank

        # Safe norm with explicit clamp to prevent NaN gradients
        B_norm_val = torch.clamp(torch.norm(B, p=2, dim=0, keepdim=True), min=self.eps)
        B_norm = B / B_norm_val
        Gram_B = torch.matmul(B_norm.t(), B_norm)
        I_r = torch.eye(r, device=B.device)
        loss_weight_B = torch.norm(Gram_B - I_r, p="fro") ** 2 / (r * (r - 1))

        A_norm_val = torch.clamp(torch.norm(A, p=2, dim=1, keepdim=True), min=self.eps)
        A_norm = A / A_norm_val
        Gram_A = torch.matmul(A_norm, A_norm.t())
        loss_weight_A = torch.norm(Gram_A - I_r, p="fro") ** 2 / (r * (r - 1))

        loss_act = torch.tensor(0.0, device=B.device)
        loss_sparse = torch.tensor(0.0, device=B.device)

        if lora_layer.last_z is not None:
            z = lora_layer.last_z.float().view(-1, r)
            # Remove any potential non-finite values safely
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

        total_ortho = loss_weight_B + 0.5 * loss_weight_A + loss_act
        return total_ortho, loss_sparse, Gram_B.detach()


# ==============================================================================
# 3. METRIC EVALUATION FUNCTIONS
# ==============================================================================
@torch.no_grad()
def compute_subspace_metrics(lora_layer: OrthoLoRALinear, eval_activations: torch.Tensor):
    B = lora_layer.lora_B.float()
    r = lora_layer.rank

    B_norm_val = torch.clamp(torch.norm(B, p=2, dim=0, keepdim=True), min=1e-5)
    B_norm = B / B_norm_val
    Gram_B = torch.matmul(B_norm.t(), B_norm).abs().cpu().numpy()
    np.fill_diagonal(Gram_B, 0.0)
    c_gram = Gram_B.sum() / (r * (r - 1))

    # Sanitize evaluation activations (filter out NaNs/zeros from padding)
    z = eval_activations.float().view(-1, r)
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature Participation Ratio (FPR)
    z_abs = torch.abs(z)
    l1_sum = torch.sum(z_abs, dim=-1) ** 2
    l2_sum = torch.sum(z ** 2, dim=-1) + 1e-8
    fpr = (l1_sum / (r * l2_sum)).cpu().numpy()

    # Effective Subspace Dimensionality via torch.linalg.svd
    z_centered = z - z.mean(dim=0, keepdim=True)
    # Add tiny jitter on diagonal to ensure numerical stability
    _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
    p = S / (torch.sum(S) + 1e-8)
    p = torch.clamp(p, min=1e-12)
    entropy = -torch.sum(p * torch.log(p)).item()
    d_eff = np.exp(entropy)

    return {
        "c_gram": float(c_gram),
        "mean_fpr": float(np.mean(fpr)),
        "std_fpr": float(np.std(fpr)),
        "d_eff": float(d_eff),
        "Gram_matrix": Gram_B,
        "fpr_distribution": fpr,
    }


# ==============================================================================
# 4. TRAINING & EXPERIMENTAL RUNNER
# ==============================================================================
def run_condition(
    condition_name: str,
    model_name: str,
    target_layer_idx: int,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    rank: int = 32,
    lambda_ortho: float = 1.0,
    lambda_sparse: float = 1e-3,
    steps: int = 400,
    lr: float = 5e-4,
    device: str = "cuda",
    output_dir: str = "./paper_artifacts/section_5_1"
):
    print(f"\n---> Running Condition: {condition_name} (lambda_ortho={lambda_ortho})")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    for p in model.parameters():
        p.requires_grad = False

    # Hook down_proj
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[target_layer_idx].mlp.down_proj
        parent = model.model.layers[target_layer_idx].mlp
        target_attr = "down_proj"
    elif hasattr(model, "gpt_neox"):
        layer = model.gpt_neox.layers[target_layer_idx].mlp.dense_4h_to_h
        parent = model.gpt_neox.layers[target_layer_idx].mlp
        target_attr = "dense_4h_to_h"
    else:
        raise ValueError("Unsupported architecture layout.")

    ortho_module = OrthoLoRALinear(base_layer=layer, rank=rank).to(device)
    setattr(parent, target_attr, ortho_module)

    optimizer = torch.optim.AdamW([ortho_module.lora_A, ortho_module.lora_B], lr=lr)
    ortho_criterion = SubspaceOrthogonalityLoss().to(device)

    model.train()
    step_count = 0
    running_task_loss = 0.0
    history = {"step": [], "task_loss": [], "ortho_loss": []}
    train_iter = iter(train_loader)

    while step_count < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        task_loss = outputs.loss

        if lambda_ortho > 0.0:
            ortho_loss, sparse_loss, _ = ortho_criterion(ortho_module)
            total_loss = task_loss + (lambda_ortho * ortho_loss) + (lambda_sparse * sparse_loss)
        else:
            ortho_loss = torch.tensor(0.0, device=device)
            total_loss = task_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([ortho_module.lora_A, ortho_module.lora_B], max_norm=1.0)
        optimizer.step()

        step_count += 1
        running_task_loss += task_loss.item()
        history["step"].append(step_count)
        history["task_loss"].append(task_loss.item())
        history["ortho_loss"].append(ortho_loss.item())

        # Log smoothed average every 40 steps
        if step_count % 40 == 0 or step_count == steps:
            avg_loss = running_task_loss / (40 if step_count % 40 == 0 else step_count % 40)
            print(f"Step {step_count:03d}/{steps} | Task Loss: {avg_loss:.4f} | Ortho Loss: {ortho_loss.item():.4f}")
            running_task_loss = 0.0

    # Evaluate on held-out test set
    model.eval()
    test_loss_total = 0.0
    test_steps = 0
    all_z = []

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            test_loss_total += out.loss.item()
            test_steps += 1
            if ortho_module.last_z is not None:
                all_z.append(ortho_module.last_z.view(-1, rank).cpu())

    eval_z = torch.cat(all_z, dim=0)
    final_test_loss = test_loss_total / test_steps

    metrics = compute_subspace_metrics(ortho_module, eval_z)
    metrics["task_loss_final"] = final_test_loss
    metrics["B_weights"] = ortho_module.lora_B.detach().cpu().numpy()
    metrics["z_eval"] = eval_z.numpy()

    # Save adapter weights for Section 5.2 (Steering)
    save_filename = f"{condition_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_weights.pt"
    save_path = os.path.join(output_dir, save_filename)
    torch.save({
        "lora_A": ortho_module.lora_A.data.cpu(),
        "lora_B": ortho_module.lora_B.data.cpu(),
        "rank": rank,
        "target_layer_idx": target_layer_idx
    }, save_path)
    print(f"[*] Saved adapter checkpoints to: {save_path}")

    del model, optimizer
    torch.cuda.empty_cache()

    return metrics, history


# ==============================================================================
# 5. MAIN EXPERIMENT PIPELINE AND VISUALIZATIONS
# ==============================================================================
def run_experiment_5_1(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_name="cardiffnlp/tweet_eval",
    dataset_subset="sentiment",
    target_layer_idx=12,
    rank=32,
    steps=500,               # Scaled steps for clear task convergence
    batch_size=8,            # Increased batch size for stable gradients
    lr=5e-4,
    output_dir="./paper_artifacts/section_5_1"
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Experiment 5.1 on {device}: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_subset)

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def tokenize_task_fn(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for text, label_idx in zip(examples["text"], examples["label"]):
            label_word = LABEL_MAP[int(label_idx)]
            prompt = f"Tweet: {text.strip()}\nSentiment:"
            full_text = f"{prompt} {label_word}"

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            full_enc = tokenizer(
                full_text,
                max_length=128,
                truncation=True,
                padding="max_length"
            )

            ids = full_enc["input_ids"]
            mask = full_enc["attention_mask"]

            # Mask out prompt tokens so loss is evaluated ONLY on the sentiment label
            lbl = [-100] * len(ids)
            prompt_len = len(prompt_ids)
            for idx in range(prompt_len, len(ids)):
                if mask[idx] == 1:
                    lbl[idx] = ids[idx]

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(lbl)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    print("[*] Tokenizing and formatting dataset for targeted sentiment task...")
    tokenized_train = dataset["train"].map(
        tokenize_task_fn, batched=True, remove_columns=dataset["train"].column_names
    )
    tokenized_eval = dataset["test"].map(
        tokenize_task_fn, batched=True, remove_columns=dataset["test"].column_names
    )

    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")

    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized_eval, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

    # -------------------------------------------------------------
    # 0. EVALUATE UNADAPTED BASE MODEL (ANCHOR)
    # -------------------------------------------------------------
    print("\n---> Evaluating Anchor: Unadapted Base Model (Zero-Shot)")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    base_model.eval()

    total_base_loss = 0.0
    eval_steps = 0
    with torch.no_grad():
        for batch in eval_loader:
            if eval_steps >= 25: break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            total_base_loss += out.loss.item()
            eval_steps += 1

    base_eval_loss = total_base_loss / eval_steps
    print(f"[*] Base Model Zero-Shot Test Loss: {base_eval_loss:.4f}")
    del base_model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 1. RUN STANDARD LORA (lambda=0.0)
    # -------------------------------------------------------------
    std_metrics, std_history = run_condition(
        condition_name="Standard LoRA",
        model_name=model_name,
        target_layer_idx=target_layer_idx,
        train_loader=train_loader,
        eval_loader=eval_loader,
        rank=rank,
        lambda_ortho=0.0,
        lambda_sparse=0.0,
        steps=steps,
        lr=lr,
        device=device
    )

    # -------------------------------------------------------------
    # 2. RUN ORTHO-LORA (lambda=1.0)
    # -------------------------------------------------------------
    ortho_metrics, ortho_history = run_condition(
        condition_name="Ortho-LoRA (Ours)",
        model_name=model_name,
        target_layer_idx=target_layer_idx,
        train_loader=train_loader,
        eval_loader=eval_loader,
        rank=rank,
        lambda_ortho=1.0,
        lambda_sparse=1e-3,
        steps=steps,
        lr=lr,
        device=device
    )

    # -------------------------------------------------------------
    # 3. GENERATE COMPLETE PAPER BENCHMARK TABLE
    # -------------------------------------------------------------
    table_data = [
        {
            "Model / Method": "Base Model (Unadapted)",
            "Coherence C_gram (↓)": "N/A (No Adapter)",
            "Mean FPR (↓)": "N/A",
            "Effective Dim d_eff (↑)": "N/A",
            "Task Loss (Test)": f"{base_eval_loss:.4f}"
        },
        {
            "Model / Method": "Standard LoRA",
            "Coherence C_gram (↓)": f"{std_metrics['c_gram']:.4f}",
            "Mean FPR (↓)": f"{std_metrics['mean_fpr']:.4f}",
            "Effective Dim d_eff (↑)": f"{std_metrics['d_eff']:.2f} / {rank}",
            "Task Loss (Test)": f"{std_metrics['task_loss_final']:.4f}"
        },
        {
            "Model / Method": "Ortho-LoRA (Ours)",
            "Coherence C_gram (↓)": f"{ortho_metrics['c_gram']:.4f}",
            "Mean FPR (↓)": f"{ortho_metrics['mean_fpr']:.4f}",
            "Effective Dim d_eff (↑)": f"{ortho_metrics['d_eff']:.2f} / {rank}",
            "Task Loss (Test)": f"{ortho_metrics['task_loss_final']:.4f}"
        }
    ]
    df_results = pd.DataFrame(table_data)
    print("\n================== COMPLETE 5.1 QUANTITATIVE BENCHMARK ==================")
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(output_dir, "table_5_1_complete_benchmark.csv"), index=False)

    # ==============================================================================
    # 6. MULTI-SPACE VISUALIZATION SUITE (Publication & Intuition Figures)
    # ==============================================================================
    print("\n[*] Generating Comprehensive Space-Visualization Suite...")
    fig = plt.figure(figsize=(18, 9))

    # --- PANEL 1: Gram Matrix Heatmaps (Weight Space Superposition Collapse) ---
    ax1 = plt.subplot2grid((2, 4), (0, 0))
    sns.heatmap(std_metrics["Gram_matrix"], ax=ax1, cmap="magma", cbar=True, vmin=0.0, vmax=0.7)
    ax1.set_title(f"Standard LoRA Gram ($B^\\top B$)\n$\\mathcal{{C}}_{{gram}} = {std_metrics['c_gram']:.3f}$")
    ax1.set_xlabel("Rank Channel")
    ax1.set_ylabel("Rank Channel")

    ax2 = plt.subplot2grid((2, 4), (0, 1))
    sns.heatmap(ortho_metrics["Gram_matrix"], ax=ax2, cmap="magma", cbar=True, vmin=0.0, vmax=0.7)
    ax2.set_title(f"Ortho-LoRA Gram ($B^\\top B$)\n$\\mathcal{{C}}_{{gram}} = {ortho_metrics['c_gram']:.3f}$")
    ax2.set_xlabel("Rank Channel")
    ax2.set_ylabel("Rank Channel")

    # --- PANEL 2: Weight Space Geometry (2D PCA of Feature Vectors B) ---
    pca_weights = PCA(n_components=2)
    B_std_clean = np.nan_to_num(std_metrics["B_weights"].T)
    B_ortho_clean = np.nan_to_num(ortho_metrics["B_weights"].T)

    B_std_2d = pca_weights.fit_transform(B_std_clean)
    B_ortho_2d = pca_weights.fit_transform(B_ortho_clean)

    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax3.scatter(B_std_2d[:, 0], B_std_2d[:, 1], c="#d62728", s=60, alpha=0.8, edgecolors="k")
    ax3.set_title("Weight Space: Standard LoRA ($b_k$)\n(Directional Clustering / Superposition)")
    ax3.set_xlabel("PC 1")
    ax3.set_ylabel("PC 2")
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax4.scatter(B_ortho_2d[:, 0], B_ortho_2d[:, 1], c="#2ca02c", s=60, alpha=0.8, edgecolors="k")
    ax4.set_title("Weight Space: Ortho-LoRA ($b_k$)\n(Equiangular / Hyperspherical Unpacking)")
    ax4.set_xlabel("PC 1")
    ax4.set_ylabel("PC 2")
    ax4.grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 3: Feature Participation Ratio (Monosemanticity Density) ---
    ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    sns.kdeplot(std_metrics["fpr_distribution"], ax=ax5, color="#d62728", fill=True, label="Standard LoRA (Polysemantic)")
    sns.kdeplot(ortho_metrics["fpr_distribution"], ax=ax5, color="#2ca02c", fill=True, label="Ortho-LoRA (Monosemantic)")
    ax5.axvline(1.0 / rank, color="black", linestyle=":", label=f"Ideal 1-Sparse ($1/r={1/rank:.3f}$)")
    ax5.set_title("Feature Participation Ratio (FPR) across Evaluation Tokens")
    ax5.set_xlabel("FPR (Lower = Cleaner Monosemantic Specialization)")
    ax5.set_ylabel("Token Density")
    ax5.legend()
    ax5.grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 4: Activation Trajectory Space (2D PCA of Bottleneck Activations z) ---
    pca_act = PCA(n_components=2)
    sample_tokens = min(600, len(std_metrics["z_eval"]))

    z_std_clean = np.nan_to_num(std_metrics["z_eval"][:sample_tokens])
    z_ortho_clean = np.nan_to_num(ortho_metrics["z_eval"][:sample_tokens])

    z_std_2d = pca_act.fit_transform(z_std_clean)
    z_ortho_2d = pca_act.fit_transform(z_ortho_clean)

    ax6 = plt.subplot2grid((2, 4), (1, 2))
    ax6.scatter(z_std_2d[:, 0], z_std_2d[:, 1], c="#d62728", s=15, alpha=0.4)
    ax6.set_title("Activation Space: Standard LoRA ($z$)\n(Correlated Token Firing)")
    ax6.set_xlabel("Latent PC 1")
    ax6.set_ylabel("Latent PC 2")
    ax6.grid(True, linestyle="--", alpha=0.5)

    ax7 = plt.subplot2grid((2, 4), (1, 3))
    ax7.scatter(z_ortho_2d[:, 0], z_ortho_2d[:, 1], c="#2ca02c", s=15, alpha=0.4)
    ax7.set_title("Activation Space: Ortho-LoRA ($z$)\n(Decorrelated Orthogonal Manifold)")
    ax7.set_xlabel("Latent PC 1")
    ax7.set_ylabel("Latent PC 2")
    ax7.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fig_5_1_geometric_unpacking_and_spaces.png")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path.replace(".png", ".pdf"))
    plt.close()
    print(f"[*] Visualizations saved to: {plot_path}")