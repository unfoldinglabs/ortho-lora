"""
========================================================================================
EXPERIMENT 6.2: ARCHITECTURAL DEPTH (EARLY VS. MID VS. LATE LAYER INTERVENTION)
========================================================================================

WHAT THIS PIPELINE EXPLORES:
This experiment investigates where in the transformer hierarchy representation superposition
is most amenable to endogenous orthogonalization. By intervening independently at three
strategic depth cuts (Early ~25%, Mid ~50%, Late ~75%), we test whether monosemanticity
is best enforced where abstract semantic features are composed (mid layers) versus where
tokens are parsed (early) or projected toward the vocabulary logits (late).

HOW IT DOES IT:
1. Targets three layer positions in Qwen/Qwen2.5-0.5B (24 layers total):
   - Early Layer: Layer 6 (25% depth)
   - Mid Layer:   Layer 12 (50% depth)
   - Late Layer:  Layer 18 (75% depth)
2. For each depth, trains twin conditions under identical budgets on cardiffnlp/tweet_eval:
   - Condition A: Standard LoRA (rank=32, lambda=0.0)
   - Condition B: Ortho-LoRA (rank=32, lambda=1.0)
3. Evaluates layer-specific behaviors across geometric, functional, and causal axes:
   - Final Downstream Task Cross-Entropy Loss.
   - Off-Diagonal Gram Coherence C_gram (Degree of Superposition Elimination).
   - Mean Feature Participation Ratio (FPR, token monosemanticity).
   - Causal Steering Selectivity Index (SSI) when steering the primary task concept rank.
4. Generates space-visualizations:
   - Layer-wise Gram Matrix heatmaps side-by-side.
   - Depth-dependent Pareto curve (Task Loss vs C_gram vs SSI).
   - 2D PCA activation trajectories comparing causal steering paths across network depths.
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
from sklearn.decomposition import PCA
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
# 1. STEERABLE ORTHO-LORA LINEAR MODULE
# ==============================================================================
class SteerableOrthoLoRA(nn.Module):
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

        self.active_steering_rank = None
        self.active_steering_gamma = 0.0
        self.last_z = None

    def set_steering(self, rank_idx: int = None, gamma: float = 0.0):
        self.active_steering_rank = int(rank_idx) if rank_idx is not None else None
        self.active_steering_gamma = float(gamma)

    def clear_steering(self):
        self.active_steering_rank = None
        self.active_steering_gamma = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_fp32 = x.to(torch.float32)
        z = F.linear(x_fp32, self.lora_A)

        if self.active_steering_rank is not None and self.active_steering_gamma != 0.0:
            steer_delta = torch.zeros_like(z)
            steer_delta[:, :, self.active_steering_rank] = float(self.active_steering_gamma)
            z = z + steer_delta

        self.last_z = z
        lora_out = F.linear(z, self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


class SubspaceOrthogonalityLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, lora_layer: SteerableOrthoLoRA):
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


# ==============================================================================
# 2. NUMERICALLY STABLE METRICS & CAUSAL STEERING EVALUATOR
# ==============================================================================
def compute_safe_kl(logits_base, logits_steered, target_token_ids):
    mask_ids = list(target_token_ids.values())
    l_base = logits_base.clone().float()
    l_steer = logits_steered.clone().float()
    l_base[..., mask_ids] = -1e9
    l_steer[..., mask_ids] = -1e9

    log_p_base = F.log_softmax(l_base, dim=-1)
    log_p_steer = F.log_softmax(l_steer, dim=-1)
    p_base = torch.exp(log_p_base)

    kl_terms = torch.where(p_base > 1e-12, p_base * (log_p_base - log_p_steer), torch.zeros_like(p_base))
    return float(torch.clamp(torch.sum(kl_terms, dim=-1), min=0.0).mean().item())


@torch.no_grad()
def evaluate_layer_causal_ssi(model, lora_module, eval_loader, target_token_ids, device, rank=32):
    model.eval()
    lora_module.clear_steering()
    pos_tok_id = target_token_ids["positive"]
    neg_tok_id = target_token_ids["negative"]

    # 1. Identify primary sentiment rank
    pos_acts, neg_acts = [], []
    for i, batch in enumerate(eval_loader):
        if i >= 25: break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model(input_ids=input_ids, attention_mask=attention_mask)
        z = lora_module.last_z
        for b in range(labels.shape[0]):
            target_pos = (labels[b] != -100).nonzero(as_tuple=True)[0]
            if len(target_pos) > 0:
                t_idx = target_pos[0].item() - 1
                target_id = labels[b, target_pos[0]].item()
                act_vec = z[b, t_idx, :].cpu().numpy()
                if target_id == pos_tok_id:
                    pos_acts.append(act_vec)
                elif target_id == neg_tok_id:
                    neg_acts.append(act_vec)

    contrast = (np.mean(pos_acts, axis=0) if pos_acts else np.zeros(rank)) - \
               (np.mean(neg_acts, axis=0) if neg_acts else np.zeros(rank))
    best_rank = int(np.argmax(np.abs(contrast)))
    sign_dir = float(np.sign(contrast[best_rank])) if contrast[best_rank] != 0.0 else 1.0

    # 2. Steer with gamma = +4.0
    gamma = 4.0 * sign_dir
    lora_module.set_steering(rank_idx=best_rank, gamma=gamma)

    prob_shifts, kl_divs, traj_pts = [], [], []
    eval_steps = 0
    for batch in eval_loader:
        if eval_steps >= 15: break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out_steered = model(input_ids=input_ids, attention_mask=attention_mask)
        l_steered = out_steered.logits
        z_steered = lora_module.last_z

        lora_module.clear_steering()
        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
        l_base = out_base.logits

        lora_module.set_steering(rank_idx=best_rank, gamma=gamma)

        for b in range(labels.shape[0]):
            target_indices = (labels[b] != -100).nonzero(as_tuple=True)[0]
            if len(target_indices) == 0: continue
            pred_idx = target_indices[0].item() - 1

            p_b = F.softmax(l_base[b, pred_idx, :], dim=-1)
            p_s = F.softmax(l_steered[b, pred_idx, :], dim=-1)

            delta_p = (p_s[pos_tok_id] - p_b[pos_tok_id]).item()
            prob_shifts.append(delta_p)

            kl_val = compute_safe_kl(
                l_base[b, pred_idx:pred_idx+1, :],
                l_steered[b, pred_idx:pred_idx+1, :],
                target_token_ids
            )
            kl_divs.append(kl_val)
            traj_pts.append(z_steered[b, pred_idx, :].cpu().numpy())

        eval_steps += 1

    lora_module.clear_steering()
    mean_delta_p = float(np.mean(prob_shifts))
    mean_kl = float(np.mean(kl_divs))
    ssi = abs(mean_delta_p) / (mean_kl + 1e-5)
    mean_traj = np.mean(traj_pts, axis=0) if traj_pts else np.zeros(rank)

    return ssi, mean_delta_p, mean_kl, best_rank, mean_traj


# ==============================================================================
# 3. LAYER EXPERIMENT RUNNER
# ==============================================================================
def run_layer_condition(
    model_name: str,
    layer_idx: int,
    lambda_ortho: float,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    target_token_ids: dict,
    rank: int = 32,
    steps: int = 250,
    lr: float = 5e-4,
    device: str = "cuda"
):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    for p in model.parameters():
        p.requires_grad = False

    target_layer = model.model.layers[layer_idx].mlp.down_proj
    parent = model.model.layers[layer_idx].mlp

    lora_module = SteerableOrthoLoRA(base_layer=target_layer, rank=rank, device=device)
    setattr(parent, "down_proj", lora_module)

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

    # Gram coherence
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

    # Causal Steering Evaluation
    ssi, delta_p, collateral_kl, best_rank, mean_traj = evaluate_layer_causal_ssi(
        model, lora_module, eval_loader, target_token_ids, device, rank=rank
    )

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "layer_idx": layer_idx,
        "lambda_ortho": lambda_ortho,
        "test_loss": float(test_loss),
        "c_gram": c_gram,
        "fpr": fpr,
        "ssi": ssi,
        "delta_p": delta_p,
        "collateral_kl": collateral_kl,
        "best_rank": best_rank,
        "Gram_matrix": Gram_B,
        "mean_traj": mean_traj
    }


# ==============================================================================
# 4. MAIN EXPERIMENT RUNNER
# ==============================================================================
def run_experiment_6_2(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_name="cardiffnlp/tweet_eval",
    dataset_subset="sentiment",
    rank=32,
    output_dir="./paper_artifacts/section_6_2"
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Experiment 6.2 (Architectural Depth Ablation) on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_subset)
    target_token_ids = {
        "negative": tokenizer.encode(" negative", add_special_tokens=False)[-1],
        "neutral":  tokenizer.encode(" neutral",  add_special_tokens=False)[-1],
        "positive": tokenizer.encode(" positive", add_special_tokens=False)[-1],
    }

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

    # 3 Strategic Layer Cuts: Early (25%), Mid (50%), Late (75%)
    layers_dict = {
        "Early (~25% Depth)": 6,
        "Mid (~50% Depth)":   12,
        "Late (~75% Depth)":  18
    }

    results = []
    gram_matrices = {}
    trajectories = {}

    for depth_name, l_idx in layers_dict.items():
        print(f"\n=======================================================")
        print(f"[*] Testing Layer Position: {depth_name} (Layer {l_idx})")
        print(f"=======================================================")

        # Condition 1: Standard LoRA
        print(f"---> Training Standard LoRA at Layer {l_idx}...")
        std_res = run_layer_condition(
            model_name=model_name, layer_idx=l_idx, lambda_ortho=0.0,
            train_loader=train_loader, eval_loader=eval_loader, target_token_ids=target_token_ids,
            rank=rank, steps=250, device=device
        )
        std_res["Depth"] = depth_name
        std_res["Method"] = "Standard LoRA"
        gram_matrices[(depth_name, "Standard")] = std_res.pop("Gram_matrix")
        trajectories[(depth_name, "Standard")] = std_res.pop("mean_traj")
        results.append(std_res)

        # Condition 2: Ortho-LoRA (Ours)
        print(f"---> Training Ortho-LoRA (Ours) at Layer {l_idx}...")
        ortho_res = run_layer_condition(
            model_name=model_name, layer_idx=l_idx, lambda_ortho=1.0,
            train_loader=train_loader, eval_loader=eval_loader, target_token_ids=target_token_ids,
            rank=rank, steps=250, device=device
        )
        ortho_res["Depth"] = depth_name
        ortho_res["Method"] = "Ortho-LoRA (Ours)"
        gram_matrices[(depth_name, "Ortho")] = ortho_res.pop("Gram_matrix")
        trajectories[(depth_name, "Ortho")] = ortho_res.pop("mean_traj")
        results.append(ortho_res)

    df_layers = pd.DataFrame(results)
    df_layers.to_csv(os.path.join(output_dir, "table_6_2_layer_ablation_benchmark.csv"), index=False)
    print("\n================== 6.2 LAYER ABLATION BENCHMARK TABLE ==================")
    cols_to_print = ["Depth", "Method", "test_loss", "c_gram", "fpr", "ssi", "delta_p", "collateral_kl"]
    print(df_layers[cols_to_print].to_string(index=False))

    # ==============================================================================
    # 5. MULTI-PANEL LAYER DEPTH VISUALIZATION SUITE
    # ==============================================================================
    print("\n[*] Generating Multi-Panel Layer Ablation Visualizations...")
    fig = plt.figure(figsize=(18, 9))

    # --- PANEL 1: Gram Matrices Across Depths (Ortho-LoRA Only) ---
    for idx, (depth_name, l_idx) in enumerate(layers_dict.items()):
        ax = plt.subplot2grid((2, 3), (0, idx))
        gram_data = gram_matrices[(depth_name, "Ortho")]
        c_val = df_layers[(df_layers["Depth"] == depth_name) & (df_layers["Method"] == "Ortho-LoRA (Ours)")]["c_gram"].values[0]
        sns.heatmap(gram_data, ax=ax, cmap="magma", cbar=True, vmin=0.0, vmax=0.5)
        ax.set_title(f"{depth_name} (Layer {l_idx})\nOrtho-LoRA $\\mathcal{{C}}_{{gram}} = {c_val:.3f}$")
        ax.set_xlabel("Rank Channel")
        ax.set_ylabel("Rank Channel")

    # --- PANEL 2: Task Loss vs Layer Depth ---
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    sns.barplot(data=df_layers, x="Depth", y="test_loss", hue="Method", ax=ax4, palette={"Standard LoRA": "#d62728", "Ortho-LoRA (Ours)": "#2ca02c"})
    ax4.set_title("(a) Downstream Task Loss vs. Layer Depth")
    ax4.set_ylabel("Test Loss (Lower is Better)")
    ax4.grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 3: Steering Selectivity Index (SSI) vs Layer Depth ---
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    sns.barplot(data=df_layers, x="Depth", y="ssi", hue="Method", ax=ax5, palette={"Standard LoRA": "#d62728", "Ortho-LoRA (Ours)": "#2ca02c"})
    ax5.set_title("(b) Causal Steering Selectivity (SSI) vs. Depth\n(Mid-Layer Peak Monosemanticity)")
    ax5.set_ylabel("Selectivity Index (SSI) (Higher is Better)")
    ax5.grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 4: 2D PCA Causal Trajectory Space across Depths ---
    ax6 = plt.subplot2grid((2, 3), (1, 2))
    pca = PCA(n_components=2)
    traj_pts = np.array(list(trajectories.values()))
    pca.fit(traj_pts)

    depth_colors = {"Early (~25% Depth)": "#1f77b4", "Mid (~50% Depth)": "#2ca02c", "Late (~75% Depth)": "#9467bd"}
    for (depth_name, method), vec in trajectories.items():
        pt_2d = pca.transform(vec.reshape(1, -1))[0]
        marker = "s" if method == "Ortho-LoRA (Ours)" else "o"
        edge = "black" if method == "Ortho-LoRA (Ours)" else "none"
        ax6.scatter(pt_2d[0], pt_2d[1], color=depth_colors[depth_name], marker=marker, s=120, edgecolors=edge,
                    label=f"{depth_name} ({method.split()[0]})")

    ax6.set_title("(c) Steered Latent Subspace Projections\nAcross Architectural Depths")
    ax6.set_xlabel("PC 1")
    ax6.set_ylabel("PC 2")
    ax6.grid(True, linestyle="--", alpha=0.5)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fig_6_2_layer_ablation.png")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path.replace(".png", ".pdf"))
    plt.close()

    print(f"[*] Visualizations cleanly saved to: {plot_path}")
    print("[*] Experiment 6.2 execution complete.")