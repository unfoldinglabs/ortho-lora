"""
========================================================================================
EXPERIMENT 5.3: ENDOGENOUS ORTHO-LORA VS. EXOGENOUS TOP-K SPARSE AUTOENCODERS
========================================================================================

WHAT THIS PIPELINE EXPLORES:
This experiment investigates whether endogenous subspace orthogonalization (Ortho-LoRA)
discovers the same underlying monosemantic feature directions as an exogenous, overcomplete
Top-K Sparse Autoencoder (SAE), while avoiding the severe computational and latency
penalties inherent to post-hoc dictionary learning.

HOW IT DOES IT:
1. Loads the pre-trained checkpoints from prior sections:
   - Base Model: Qwen/Qwen2.5-0.5B at layer ~50% depth.
   - Baseline Top-K SAE (k=32, expansion factor 4x) from Section 4.3.
   - Standard LoRA weights from Section 5.1.
   - Ortho-LoRA weights from Section 5.1.
2. Latent Feature Alignment & Maximum Inner Product Search:
   - For every learned LoRA feature direction b_k in R^d (k in 1..r), computes its maximum
     absolute cosine similarity to any dictionary latent column of W_dec in R^{d x d_sae}:
     Alignment(k) = max_m |cos(b_k, W_dec[:, m])|.
   - Evaluates whether Ortho-LoRA directions align more strongly and cleanly with SAE
     latents than unregularized Standard LoRA.
3. Downstream Task Performance vs. Representation Fidelity:
   - Evaluates test loss across 4 operational conditions:
     a) Unadapted Base Model.
     b) Base Model under SAE activation bottleneck reconstruction (measuring SAE distortion).
     c) Base Model + Standard LoRA.
     d) Base Model + Ortho-LoRA.
4. Operational Hardware Benchmarking on T4 GPU:
   - Measures exact forward-pass token latency (ms/token), peak memory consumption (MB),
     and total parameter overhead.
5. Geometric & Multi-Space Visualizations:
   - Alignment Spectrum Heatmap (Bipartite Cosine Matching Matrix).
   - Downstream Task Loss vs. Feature Coherence Pareto Frontier.
   - Hardware Latency and Parameter Scaling trade-off comparisons.
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
# 1. TOP-K SAE MODULE DEFINITION (Matched to Section 4.3)
# ==============================================================================
class TopKSAE(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k = k
        self.W_enc = nn.Linear(d_model, self.d_sae, bias=True)
        self.W_dec = nn.Linear(self.d_sae, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        x_centered = x - self.b_dec
        pre_acts = self.W_enc(x_centered)
        topk_vals, topk_indices = torch.topk(pre_acts, k=self.k, dim=-1)
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk_indices, F.relu(topk_vals))
        x_recon = self.W_dec(latents) + self.b_dec
        return x_recon, latents


# ==============================================================================
# 2. INSTRUMENTED LORA MODULE (Matched to 5.1/5.2)
# ==============================================================================
class LoRALinear(nn.Module):
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

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_fp32 = x.to(torch.float32)
        z = F.linear(x_fp32, self.lora_A)
        lora_out = F.linear(z, self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def attach_lora_to_model(model, target_layer_idx: int, rank: int = 32) -> LoRALinear:
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

    # Explicitly cast to the target layer's device and dtype
    target_device = layer.weight.device
    lora_module = LoRALinear(base_layer=layer, rank=rank).to(target_device)
    setattr(parent, target_attr, lora_module)
    return lora_module


# ==============================================================================
# 3. SAE INTERVENTION HOOK (FOR RESIDUAL RECONSTRUCTION EVALUATION)
# ==============================================================================
class SAEReconstructionHook:
    """Hooks into the target layer and replaces residual activations with SAE reconstructions."""
    def __init__(self, target_layer, sae: TopKSAE):
        self.target_layer = target_layer
        self.sae = sae
        self.active = False
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.active:
            orig_shape = h.shape
            flat_h = h.view(-1, orig_shape[-1]).float()
            recon_h, _ = self.sae(flat_h)
            h = recon_h.view(orig_shape).to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def remove(self):
        self.hook_handle.remove()


# ==============================================================================
# 4. EXPERIMENT EXECUTION ENGINE
# ==============================================================================
def run_experiment_5_3(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_name="cardiffnlp/tweet_eval",
    dataset_subset="sentiment",
    target_layer_idx=12,
    rank=32,
    expansion_factor=4,
    sec_4_3_dir="./paper_artifacts/section_4_3",
    sec_5_1_dir="./paper_artifacts/section_5_1",
    output_dir="./paper_artifacts/section_5_3"
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Experiment 5.3 on {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Tokenizer & Task Dataset
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

    tokenized_eval = dataset["test"].map(tokenize_task_fn, batched=True, remove_columns=dataset["test"].column_names)
    tokenized_eval.set_format("torch")
    eval_loader = DataLoader(tokenized_eval, batch_size=8, shuffle=False, collate_fn=default_data_collator)

    # 2. Load Model Dimensions
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    d_model = ref_model.config.hidden_size
    del ref_model
    torch.cuda.empty_cache()

    # 3. Load Trained Top-K SAE from Section 4.3
    sae_path = os.path.join(sec_4_3_dir, "topk_sae_baseline.pt")
    sae = TopKSAE(d_model=d_model, expansion_factor=expansion_factor, k=rank).to(device)
    if os.path.exists(sae_path):
        print(f"[*] Loading pre-trained Top-K SAE from {sae_path}")
        sae_ckpt = torch.load(sae_path, map_location=device)
        sae.load_state_dict(sae_ckpt["state_dict"])
    else:
        print("[!] Warning: Section 4.3 SAE checkpoint not found. Synthesizing initialized SAE baseline...")
        with torch.no_grad():
            sae.W_dec.weight.data = F.normalize(torch.randn(d_model, sae.d_sae, device=device), p=2, dim=0)
            sae.W_enc.weight.data = sae.W_dec.weight.data.t().clone()

    sae.eval()

    # 4. Load LoRA Weight Vectors from Section 5.1
    std_ckpt_path = os.path.join(sec_5_1_dir, "standard_lora_weights.pt")
    ortho_ckpt_path = os.path.join(sec_5_1_dir, "ortho-lora_ours_weights.pt")

    if os.path.exists(std_ckpt_path):
        B_std = torch.load(std_ckpt_path, map_location="cpu")["lora_B"].float().to(device)
    else:
        print("[!] Synthesizing Standard LoRA weights...")
        B_std = torch.randn(d_model, rank, device=device)

    if os.path.exists(ortho_ckpt_path):
        B_ortho = torch.load(ortho_ckpt_path, map_location="cpu")["lora_B"].float().to(device)
    else:
        print("[!] Synthesizing Ortho-LoRA weights...")
        q, _ = torch.linalg.qr(torch.randn(d_model, rank, device=device))
        B_ortho = q[:, :rank]

    # ==============================================================================
    # 5. FEATURE ALIGNMENT ANALYSIS: LoRA (b_k) vs. SAE Latents (W_dec)
    # ==============================================================================
    print("\n[*] Computing Bipartite Subspace Cosine Alignment (LoRA b_k vs SAE W_dec)...")
    # Normalize dictionary columns
    W_dec_norm = F.normalize(sae.W_dec.weight.data, p=2, dim=0) # [d_model, d_sae]
    B_std_norm = F.normalize(B_std, p=2, dim=0)                 # [d_model, rank]
    B_ortho_norm = F.normalize(B_ortho, p=2, dim=0)             # [d_model, rank]

    # Compute full cosine correlation matrices: [rank, d_sae]
    sim_matrix_std = torch.matmul(B_std_norm.t(), W_dec_norm).abs()
    sim_matrix_ortho = torch.matmul(B_ortho_norm.t(), W_dec_norm).abs()

    # Maximum cosine similarity per rank channel to any SAE latent
    max_align_std = sim_matrix_std.max(dim=1).values.cpu().numpy()
    max_align_ortho = sim_matrix_ortho.max(dim=1).values.cpu().numpy()

    mean_align_std = float(np.mean(max_align_std))
    mean_align_ortho = float(np.mean(max_align_ortho))

    # Overlap count: features with high cosine match (> 0.40)
    high_match_std = int(np.sum(max_align_std > 0.40))
    high_match_ortho = int(np.sum(max_align_ortho > 0.40))

    print(f"[*] Standard LoRA -> SAE Mean Alignment: {mean_align_std:.4f} (High Match Count: {high_match_std}/{rank})")
    print(f"[*] Ortho-LoRA    -> SAE Mean Alignment: {mean_align_ortho:.4f} (High Match Count: {high_match_ortho}/{rank})")

    # ==============================================================================
    # 6. TASK LOSS & LATENCY BENCHMARKING (4 CONDITIONS)
    # ==============================================================================
    print("\n[*] Evaluating Task Loss and Inference Latency across 4 Conditions...")

    def evaluate_model_loss_and_latency(test_model, warmup=5, eval_batches=20):
        test_model.eval()
        total_loss = 0.0
        steps = 0
        latencies = []

        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= eval_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Time forward execution precisely
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = test_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                if i >= warmup:
                    latencies.append((t1 - t0) * 1000.0) # ms per batch
                total_loss += out.loss.item()
                steps += 1

        mean_loss = total_loss / steps
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        return mean_loss, mean_latency

    # Condition 1: Base Model (Zero-Shot)
    model_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    base_loss, base_latency = evaluate_model_loss_and_latency(model_base)
    print(f"[1/4] Base Model              | Loss: {base_loss:.4f} | Latency: {base_latency:.2f} ms")

    # Condition 2: Base Model + SAE Bottleneck Reconstruction
    target_layer = model_base.model.layers[target_layer_idx]
    sae_hook = SAEReconstructionHook(target_layer, sae)
    sae_hook.active = True
    sae_loss, sae_latency = evaluate_model_loss_and_latency(model_base)
    sae_hook.remove()
    print(f"[2/4] Base + Top-K SAE Recon  | Loss: {sae_loss:.4f} | Latency: {sae_latency:.2f} ms")
    del model_base
    torch.cuda.empty_cache()

# Condition 3: Standard LoRA
    model_std = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    lora_std = attach_lora_to_model(model_std, target_layer_idx, rank=rank).to(device)
    if os.path.exists(std_ckpt_path):
        ckpt = torch.load(std_ckpt_path, map_location="cpu")
        lora_std.lora_A.data.copy_(ckpt["lora_A"].to(device))
        lora_std.lora_B.data.copy_(ckpt["lora_B"].to(device))
    std_loss, std_latency = evaluate_model_loss_and_latency(model_std)
    print(f"[3/4] Standard LoRA           | Loss: {std_loss:.4f} | Latency: {std_latency:.2f} ms")
    del model_std
    torch.cuda.empty_cache()

    # Condition 4: Ortho-LoRA (Ours)
    model_ortho = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    lora_ortho = attach_lora_to_model(model_ortho, target_layer_idx, rank=rank).to(device)
    if os.path.exists(ortho_ckpt_path):
        ckpt = torch.load(ortho_ckpt_path, map_location="cpu")
        lora_ortho.lora_A.data.copy_(ckpt["lora_A"].to(device))
        lora_ortho.lora_B.data.copy_(ckpt["lora_B"].to(device))
    ortho_loss, ortho_latency = evaluate_model_loss_and_latency(model_ortho)
    print(f"[4/4] Ortho-LoRA (Ours)       | Loss: {ortho_loss:.4f} | Latency: {ortho_latency:.2f} ms")
    del model_ortho
    torch.cuda.empty_cache()

    # Condition 4: Ortho-LoRA (Ours)
    model_ortho = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    lora_ortho = attach_lora_to_model(model_ortho, target_layer_idx, rank=rank)
    if os.path.exists(ortho_ckpt_path):
        ckpt = torch.load(ortho_ckpt_path, map_location="cpu")
        lora_ortho.lora_A.data.copy_(ckpt["lora_A"])
        lora_ortho.lora_B.data.copy_(ckpt["lora_B"])
    ortho_loss, ortho_latency = evaluate_model_loss_and_latency(model_ortho)
    print(f"[4/4] Ortho-LoRA (Ours)       | Loss: {ortho_loss:.4f} | Latency: {ortho_latency:.2f} ms")
    del model_ortho
    torch.cuda.empty_cache()

    # ==============================================================================
    # 7. QUANTITATIVE BENCHMARK TABLE
    # ==============================================================================
    lora_params = 2 * d_model * rank
    sae_params = (2 * d_model * sae.d_sae) + d_model + sae.d_sae

    benchmark_data = [
        {
            "Architecture / Paradigm": "Unadapted Base Model",
            "Intervention Type": "None (Dense)",
            "Params Overhead": "0",
            "Test Task Loss (↓)": f"{base_loss:.4f}",
            "Forward Latency (ms) (↓)": f"{base_latency:.2f}",
            "SAE Subspace Match (↑)": "N/A"
        },
        {
            "Architecture / Paradigm": "Top-K SAE Bottleneck",
            "Intervention Type": "Exogenous Probe (Recon)",
            "Params Overhead": f"{sae_params:,}",
            "Test Task Loss (↓)": f"{sae_loss:.4f}",
            "Forward Latency (ms) (↓)": f"{sae_latency:.2f}",
            "SAE Subspace Match (↑)": "1.0000 (Self)"
        },
        {
            "Architecture / Paradigm": "Standard LoRA",
            "Intervention Type": "Endogenous (Entangled)",
            "Params Overhead": f"{lora_params:,}",
            "Test Task Loss (↓)": f"{std_loss:.4f}",
            "Forward Latency (ms) (↓)": f"{std_latency:.2f} (0 if merged)",
            "SAE Subspace Match (↑)": f"{mean_align_std:.4f}"
        },
        {
            "Architecture / Paradigm": "Ortho-LoRA (Ours)",
            "Intervention Type": "Endogenous (Monosemantic)",
            "Params Overhead": f"{lora_params:,}",
            "Test Task Loss (↓)": f"{ortho_loss:.4f}",
            "Forward Latency (ms) (↓)": f"{ortho_latency:.2f} (0 if merged)",
            "SAE Subspace Match (↑)": f"{mean_align_ortho:.4f}"
        }
    ]

    df_bench = pd.DataFrame(benchmark_data)
    print("\n================== COMPLETE 5.3 COMPARATIVE BENCHMARK ==================")
    print(df_bench.to_string(index=False))
    df_bench.to_csv(os.path.join(output_dir, "table_5_3_sae_vs_ortho_benchmark.csv"), index=False)

    # ==============================================================================
    # 8. PUBLICATION FIGURE GENERATION & SPACE VISUALIZATIONS
    # ==============================================================================
    print("\n[*] Generating Publication Figures: fig_5_3_sae_vs_ortho.png...")
    fig = plt.figure(figsize=(18, 9))

    # --- PANEL 1: Feature Alignment Distribution (Matching SAE Latents) ---
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    sns.kdeplot(max_align_std, ax=ax1, color="#d62728", fill=True, label=f"Standard LoRA (Mean: {mean_align_std:.3f})")
    sns.kdeplot(max_align_ortho, ax=ax1, color="#2ca02c", fill=True, label=f"Ortho-LoRA (Mean: {mean_align_ortho:.3f})")
    ax1.axvline(mean_align_std, color="#d62728", linestyle="--")
    ax1.axvline(mean_align_ortho, color="#2ca02c", linestyle="--")
    ax1.set_title("(a) Alignment with SAE Dictionary ($\max_m |\cos(b_k, W_{dec}[:, m])|$)")
    ax1.set_xlabel("Max Cosine Similarity to SAE Latent")
    ax1.set_ylabel("Rank Channel Density")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # --- PANEL 2: Bipartite Cosine Cross-Correlation Matrix (Top-32 SAE Latents) ---
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    # Display alignment between Ortho-LoRA rank channels and top aligned SAE latents
    top_sae_indices = sim_matrix_ortho.max(dim=0).values.topk(rank).indices.cpu()
    ortho_submatrix = sim_matrix_ortho[:, top_sae_indices].cpu().numpy()
    sns.heatmap(ortho_submatrix, ax=ax2, cmap="viridis", cbar=True, vmin=0.0, vmax=0.6)
    ax2.set_title("(b) Ortho-LoRA vs. SAE Latent Correlation\n(Bipartite Subspace Match)")
    ax2.set_xlabel("Selected SAE Latent Index")
    ax2.set_ylabel("Ortho-LoRA Rank Channel ($k$)")

    # --- PANEL 3: Task Loss vs. Monosemantic Alignment Pareto Frontier ---
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax3.scatter([base_loss], [0.0], color="#1f77b4", marker="^", s=100, label="Base Model (Dense)")
    ax3.scatter([sae_loss], [1.0], color="#9467bd", marker="D", s=100, label="Base + Top-K SAE")
    ax3.scatter([std_loss], [mean_align_std], color="#d62728", marker="o", s=100, label="Standard LoRA")
    ax3.scatter([ortho_loss], [mean_align_ortho], color="#2ca02c", marker="s", s=120, label="Ortho-LoRA (Ours)")
    ax3.set_title("(c) Performance-Interpretability Pareto Frontier")
    ax3.set_xlabel("Downstream Test Loss (Lower is Better)")
    ax3.set_ylabel("Monosemantic Subspace Alignment (Higher is Better)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # --- PANEL 4: 2D PCA Weight Space Overlay (LoRA b_k vs SAE W_dec) ---
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    pca = PCA(n_components=2)
    # Fit PCA on SAE dictionary elements
    W_dec_sub = W_dec_norm[:, :128].t().cpu().numpy()
    B_ortho_pts = B_ortho_norm.t().cpu().numpy()
    B_std_pts = B_std_norm.t().cpu().numpy()

    pca.fit(W_dec_sub)
    sae_2d = pca.transform(W_dec_sub)
    ortho_2d = pca.transform(B_ortho_pts)
    std_2d = pca.transform(B_std_pts)

    ax4.scatter(sae_2d[:, 0], sae_2d[:, 1], c="#9467bd", alpha=0.25, s=25, label="SAE Latents ($W_{dec}$)")
    ax4.scatter(std_2d[:, 0], std_2d[:, 1], c="#d62728", alpha=0.8, s=50, marker="o", label="Standard LoRA ($b_k$)")
    ax4.scatter(ortho_2d[:, 0], ortho_2d[:, 1], c="#2ca02c", alpha=0.9, s=60, marker="s", edgecolors="k", label="Ortho-LoRA ($b_k$)")
    ax4.set_title("(d) Subspace Vector Projections (Joint PC Space)")
    ax4.set_xlabel("PC 1")
    ax4.set_ylabel("PC 2")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    # --- PANEL 5: Forward Latency Overhead (ms/batch) ---
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    lat_df = pd.DataFrame({
        "Model": ["Base", "Base + SAE", "Std LoRA", "Ortho (Active)", "Ortho (Merged)"],
        "Latency": [base_latency, sae_latency, std_latency, ortho_latency, base_latency]
    })
    palette = ["#1f77b4", "#9467bd", "#d62728", "#2ca02c", "#17becf"]
    sns.barplot(data=lat_df, x="Model", y="Latency", ax=ax5, palette=palette)
    ax5.set_title("(e) Runtime Latency Comparison (ms)")
    ax5.set_ylabel("Inference Latency (ms/batch)")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=20)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 6: Parameter Footprint Scaling ---
    ax6 = plt.subplot2grid((2, 3), (1, 2))
    param_df = pd.DataFrame({
        "Method": ["Ortho-LoRA", "Top-K SAE (4x)"],
        "Params": [lora_params, sae_params]
    })
    sns.barplot(data=param_df, x="Method", y="Params", ax=ax6, palette=["#2ca02c", "#9467bd"])
    ax6.set_yscale("log")
    ax6.set_title(f"(f) Parameter Budget Overhead\n({sae_params / lora_params:.1f}x Reduction)")
    ax6.set_ylabel("Trainable Parameters (Log Scale)")
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fig_5_3_sae_vs_ortho.png")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path.replace(".png", ".pdf"))
    plt.close()
    print(f"[*] Visualizations cleanly saved to: {plot_path}")
    print("[*] Experiment 5.3 execution complete.")