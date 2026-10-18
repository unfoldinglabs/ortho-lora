"""
========================================================================================
EXPERIMENT 6.1: CAPACITY THRESHOLDS & THE ORTHOGONALITY PHASE TRANSITION
========================================================================================

WHAT THIS PIPELINE EXPLORES:
This experiment investigates the empirical boundary conditions of the Johnson-Lindenstrauss
lemma within low-rank adaptation. Specifically, we test the theoretical prediction made in
Section 3.2: that eliminating superposition requires an intrinsic subspace capacity
threshold (r >= K_task). When capacity is sufficient, orthogonalization succeeds with zero
task degradation; when capacity is starved (r < K_task) or over-penalized (lambda >> 1),
forcing orthogonality induces a sharp "Phase Transition" resulting in task performance collapse.

HOW IT DOES IT:
1. Executes a systematic 2D parameter grid sweep over:
   - Subspace Rank Capacity: r in {4, 8, 16, 32}
   - Orthogonality Regularization Penalty: lambda_ortho in {0.0, 0.1, 1.0, 10.0}
2. For each of the 16 configurations, trains an OrthoLoRALinear module on Qwen/Qwen2.5-0.5B
   (at mid-layer ~50% depth) on the benchmark task (cardiffnlp/tweet_eval, "sentiment")
   for a controlled, reproducible step budget on a T4 GPU.
3. Evaluates each configuration along four geometric and performance coordinates:
   - Final Downstream Task Cross-Entropy Loss (Task Retention).
   - Off-Diagonal Gram Coherence C_gram (Degree of Superposition Collapse).
   - Mean Feature Participation Ratio (FPR, Token-Level Monosemanticity).
   - Spectral Subspace Entropy (Effective Dimensionality Utilization d_eff).
4. Synthesizes a formal 2D Phase Diagram delineating the 4 empirical regimes:
   - Regime I: Superposition / Dense (Low lambda, any r) -> Low Loss, High C_gram.
   - Regime II: Optimal Monosemanticity (Balanced lambda, Adequate r) -> Low Loss, Low C_gram.
   - Regime III: Capacity Starvation Collapse (High lambda, Low r) -> Severe Task Degradation.
   - Regime IV: Over-Regularized (Extreme lambda) -> Orthogonality smothers task learning.
5. Generates the Pareto Frontier and singular value spectra across capacity boundaries.
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
# 1. ORTHO-LORA LINEAR MODULE
# ==============================================================================
class OrthoLoRALinear(nn.Module):
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

        # Trainable matrices initialized directly in FP32 on target device
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


# ==============================================================================
# 2. ORTHOGONALITY CRITERION
# ==============================================================================
class SubspaceOrthogonalityLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, lora_layer: OrthoLoRALinear):
        B = lora_layer.lora_B.float()
        r = lora_layer.rank
        if r <= 1:
            return torch.tensor(0.0, device=B.device), torch.tensor(0.0, device=B.device)

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

        total_ortho = loss_weight_B + loss_act
        return total_ortho, loss_sparse


# ==============================================================================
# 3. METRIC EVALUATOR
# ==============================================================================
@torch.no_grad()
def evaluate_grid_point(lora_layer: OrthoLoRALinear, eval_z: torch.Tensor):
    B = lora_layer.lora_B.float()
    r = lora_layer.rank

    # Gram coherence
    B_norm_val = torch.clamp(torch.norm(B, p=2, dim=0, keepdim=True), min=1e-5)
    B_norm = B / B_norm_val
    Gram_B = torch.matmul(B_norm.t(), B_norm).abs().cpu().numpy()
    np.fill_diagonal(Gram_B, 0.0)
    c_gram = float(Gram_B.sum() / (r * (r - 1))) if r > 1 else 0.0

    # Feature Participation Ratio (FPR)
    z = eval_z.float().view(-1, r)
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z_abs = torch.abs(z)
    l1_sum = torch.sum(z_abs, dim=-1) ** 2
    l2_sum = torch.sum(z ** 2, dim=-1) + 1e-8
    fpr = (l1_sum / (r * l2_sum)).cpu().numpy()
    mean_fpr = float(np.mean(fpr))

    # Singular value spectrum & Spectral Entropy
    z_centered = z - z.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
    p = S / (torch.sum(S) + 1e-8)
    p = torch.clamp(p, min=1e-12)
    entropy = -torch.sum(p * torch.log(p)).item()
    d_eff = float(np.exp(entropy))

    return c_gram, mean_fpr, d_eff, S.cpu().numpy()


# ==============================================================================
# 4. SINGLE GRID-POINT RUNNER
# ==============================================================================
def train_single_grid_cell(
    model_name: str,
    target_layer_idx: int,
    rank: int,
    lambda_ortho: float,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    steps: int = 150,
    lr: float = 5e-4,
    device: str = "cuda"
):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    for p in model.parameters():
        p.requires_grad = False

    target_layer = model.model.layers[target_layer_idx].mlp.down_proj
    parent = model.model.layers[target_layer_idx].mlp

    # Instantiate directly on device
    ortho_module = OrthoLoRALinear(base_layer=target_layer, rank=rank, device=device)
    setattr(parent, "down_proj", ortho_module)

    optimizer = torch.optim.AdamW([ortho_module.lora_A, ortho_module.lora_B], lr=lr)
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

        if lambda_ortho > 0.0 and rank > 1:
            ortho_loss, sparse_loss = ortho_criterion(ortho_module)
            total_loss = task_loss + (lambda_ortho * ortho_loss) + (1e-3 * sparse_loss)
        else:
            total_loss = task_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([ortho_module.lora_A, ortho_module.lora_B], max_norm=1.0)
        optimizer.step()
        step_count += 1

    # Evaluation on test set
    model.eval()
    test_loss_sum = 0.0
    eval_steps = 0
    all_z = []

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 15:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            test_loss_sum += out.loss.item()
            eval_steps += 1
            if ortho_module.last_z is not None:
                all_z.append(ortho_module.last_z.view(-1, rank).cpu())

    final_loss = test_loss_sum / eval_steps
    eval_z = torch.cat(all_z, dim=0)

    c_gram, mean_fpr, d_eff, singular_vals = evaluate_grid_point(ortho_module, eval_z)

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "rank": rank,
        "lambda_ortho": lambda_ortho,
        "test_loss": float(final_loss),
        "c_gram": c_gram,
        "mean_fpr": mean_fpr,
        "d_eff": d_eff,
        "singular_values": singular_vals
    }


# ==============================================================================
# 5. MAIN EXPERIMENT EXECUTION (GRID SWEEP)
# ==============================================================================
def run_experiment_6_1(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_name="cardiffnlp/tweet_eval",
    dataset_subset="sentiment",
    target_layer_idx=12,
    output_dir="./paper_artifacts/section_6_1"
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Experiment 6.1 (Capacity Phase Transition) on {device}")

    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_subset)
    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def tokenize_task_fn(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for text, label_idx in zip(examples["text"], examples["label"]):
            prompt = f"Tweet: {text.strip()}\nSentiment:"
            full_text = f"{prompt} {LABEL_MAP[int(label_idx)]}"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            full_enc = tokenizer(full_text, max_length=128, truncation=True, padding="max_length")

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

    tokenized_train = dataset["train"].map(tokenize_task_fn, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_eval = dataset["test"].map(tokenize_task_fn, batched=True, remove_columns=dataset["test"].column_names)
    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")

    train_loader = DataLoader(tokenized_train, batch_size=8, shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized_eval, batch_size=8, shuffle=False, collate_fn=default_data_collator)

    # Define the 4x4 parameter grid
    ranks = [4, 8, 16, 32]
    lambdas = [0.0, 0.1, 1.0, 10.0]

    grid_results = []
    svd_spectra = {}

    total_cells = len(ranks) * len(lambdas)
    cell_idx = 1

    print(f"\n[*] Starting {total_cells}-Point Grid Sweep (Ranks: {ranks} x Lambdas: {lambdas})...")
    for r in ranks:
        for lam in lambdas:
            t0 = time.time()
            print(f"\n[{cell_idx}/{total_cells}] Training Cell: Rank = {r:02d} | Lambda_ortho = {lam:.1f} ...")
            cell_data = train_single_grid_cell(
                model_name=model_name,
                target_layer_idx=target_layer_idx,
                rank=r,
                lambda_ortho=lam,
                train_loader=train_loader,
                eval_loader=eval_loader,
                steps=150,
                device=device
            )
            elapsed = time.time() - t0
            print(f"    --> Loss: {cell_data['test_loss']:.4f} | C_gram: {cell_data['c_gram']:.4f} | FPR: {cell_data['mean_fpr']:.3f} | Elapsed: {elapsed:.1f}s")

            svd_spectra[(r, lam)] = cell_data.pop("singular_values")
            grid_results.append(cell_data)
            cell_idx += 1

    df_grid = pd.DataFrame(grid_results)
    df_grid.to_csv(os.path.join(output_dir, "table_6_1_phase_transition_grid.csv"), index=False)
    print("\n================== 6.1 PHASE TRANSITION GRID RESULTS ==================")
    print(df_grid.to_string(index=False))

    # ==============================================================================
    # 6. PUBLICATION VISUALIZATIONS & PHASE DIAGRAMS
    # ==============================================================================
    print("\n[*] Generating Multi-Panel Phase Diagram Suite...")
    fig = plt.figure(figsize=(18, 9))

    # Pivot tables for 2D Heatmaps
    pivot_loss = df_grid.pivot(index="rank", columns="lambda_ortho", values="test_loss")
    pivot_cgram = df_grid.pivot(index="rank", columns="lambda_ortho", values="c_gram")

    # --- PANEL 1: Task Cross-Entropy Loss Heatmap ---
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    sns.heatmap(pivot_loss, ax=ax1, cmap="YlOrRd", annot=True, fmt=".3f", cbar_kws={"label": "Test Task Loss"})
    ax1.set_title("(a) Downstream Task Cross-Entropy Loss\n(Capacity Collapse at Low $r$, High $\lambda$)")
    ax1.set_xlabel(r"Orthogonality Penalty ($\lambda_{ortho}$)")
    ax1.set_ylabel(r"Subspace Rank Capacity ($r$)")

    # --- PANEL 2: Gram Coherence C_gram Heatmap ---
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    sns.heatmap(pivot_cgram, ax=ax2, cmap="magma_r", annot=True, fmt=".3f", cbar_kws={"label": r"Coherence $\mathcal{C}_{gram}$"})
    ax2.set_title(r"(b) Gram Off-Diagonal Coherence ($\mathcal{C}_{gram}$)" + "\n(Superposition Elimination Boundary)")
    ax2.set_xlabel(r"Orthogonality Penalty ($\lambda_{ortho}$)")
    ax2.set_ylabel(r"Subspace Rank Capacity ($r$)")

    # --- PANEL 3: 2D Operational Phase Diagram ---
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    # Classify each point into one of four regimes:
    # 1. Superposition (C_gram >= 0.20, Loss < 1.0)
    # 2. Optimal Monosemanticity (C_gram < 0.20, Loss < 1.0)
    # 3. Capacity Starved Collapse (Loss >= 1.0)
    regime_matrix = np.zeros((len(ranks), len(lambdas)))
    for i, r_val in enumerate(ranks):
        for j, lam_val in enumerate(lambdas):
            sub = df_grid[(df_grid["rank"] == r_val) & (df_grid["lambda_ortho"] == lam_val)].iloc[0]
            if sub["test_loss"] >= 1.0:
                regime_matrix[i, j] = 3 # Collapse
            elif sub["c_gram"] >= 0.20:
                regime_matrix[i, j] = 1 # Superposition
            else:
                regime_matrix[i, j] = 2 # Optimal Monosemantic

    cmap_regime = sns.color_palette(["#d62728", "#2ca02c", "#7f7f7f"], as_cmap=True)
    sns.heatmap(regime_matrix, ax=ax3, cmap=cmap_regime, cbar=False, annot=False)
    ax3.set_xticklabels(lambdas)
    ax3.set_yticklabels(ranks)
    ax3.set_title("(c) Operational Phase Diagram\nGreen: Optimal | Red: Superposition | Grey: Collapse")
    ax3.set_xlabel(r"Orthogonality Penalty ($\lambda_{ortho}$)")
    ax3.set_ylabel(r"Subspace Rank Capacity ($r$)")

    # --- PANEL 4: Pareto Frontier (Task Loss vs. C_gram) ---
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    palette_ranks = {4: "#d62728", 8: "#ff7f0e", 16: "#1f77b4", 32: "#2ca02c"}
    for r in ranks:
        sub = df_grid[df_grid["rank"] == r].sort_values("lambda_ortho")
        ax4.plot(sub["c_gram"], sub["test_loss"], "o-", color=palette_ranks[r], lw=2, label=f"Rank $r={r}$")

    ax4.axhline(1.0, color="gray", linestyle=":", label="Degradation Threshold")
    ax4.set_title(r"(d) Capacity-Orthogonality Pareto Curves" + "\n(Sweet Spot: Lower Left)")
    ax4.set_xlabel(r"Off-Diagonal Coherence $\mathcal{C}_{gram}$ (Lower = Monosemantic)")
    ax4.set_ylabel("Downstream Test Loss (Lower = Better)")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    # --- PANEL 5: Effective Subspace Dimension (d_eff / r Utilization) ---
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    for lam in lambdas:
        sub = df_grid[df_grid["lambda_ortho"] == lam].sort_values("rank")
        ax5.plot(sub["rank"], sub["d_eff"], "s--", lw=2, label=rf"$\lambda={lam}$")

    ax5.plot(ranks, ranks, "k:", label="Ideal Full Rank ($d_{eff} = r$)")
    ax5.set_title(r"(e) Effective Subspace Dimension ($d_{eff}$)" + "\nvs. Physical Rank")
    ax5.set_xlabel("Physical Rank Capacity ($r$)")
    ax5.set_ylabel(r"Effective Dimension ($d_{eff}$)")
    ax5.grid(True, linestyle="--", alpha=0.5)
    ax5.legend()

    # --- PANEL 6: Singular Value Spectrum (Collapse vs. Balanced) ---
    ax6 = plt.subplot2grid((2, 3), (1, 2))
    # Contrast r=32: lambda=0.0 vs lambda=1.0 vs lambda=10.0
    sv_std = svd_spectra[(32, 0.0)]
    sv_opt = svd_spectra[(32, 1.0)]
    sv_ext = svd_spectra[(32, 10.0)]

    ax6.plot(range(1, 33), sv_std / sv_std[0], "o--", color="#d62728", label=r"$\lambda=0.0$ (Steep Decay / Superposition)")
    ax6.plot(range(1, 33), sv_opt / sv_opt[0], "s-", color="#2ca02c", label=r"$\lambda=1.0$ (Uniform Eigen-Spectrum)")
    ax6.plot(range(1, 33), sv_ext / sv_ext[0], "^:", color="#7f7f7f", label=r"$\lambda=10.0$ (Over-regularized)")
    ax6.set_title(r"(f) Singular Value Eigen-Spectrum ($r=32$)" + "\nNormalized $\sigma_i / \sigma_1$")
    ax6.set_xlabel("Singular Value Index")
    ax6.set_ylabel(r"Normalized Magnitude")
    ax6.set_yscale("log")
    ax6.grid(True, linestyle="--", alpha=0.5)
    ax6.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fig_6_1_capacity_phase_transition.png")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path.replace(".png", ".pdf"))
    plt.close()

    print(f"[*] Visualizations cleanly saved to: {plot_path}")
    print("[*] Experiment 6.1 execution complete.")