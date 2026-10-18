# Endogenous Subspace Orthogonalization: Targeted Monosemanticity via Low-Rank Adaptation (Ortho-LoRA)

[![Paper PDF](https://img.shields.io/badge/Paper-LaTeX-red.svg)](paper/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20(16GB)-green.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete experimental pipeline, analysis scripts, reproduction benchmarks, and LaTeX paper source for:
> **"Endogenous Subspace Orthogonalization: Targeted Monosemanticity via Low-Rank Adaptation"**

---

## 🔬 Research Overview

Large Language Models (LLMs) rely on **superposition**—packing more semantic features into representations than ambient dimensions by allowing small, non-orthogonal overlaps ($\langle f_i, f_j \rangle \neq 0$). While efficient, superposition creates **polysemanticity**, making features entangled, difficult to interpret, and susceptible to collateral damage when steered.

Existing interpretability methods predominantly rely on **Sparse Autoencoders (SAEs)**, which observe activations *exogenously*. SAEs incur severe parameter overhead ($4\times$ to $32\times$ layer dimension), add forward latency, and catastrophic task degradation when used interventionally.

### The Ortho-LoRA Hypothesis
For any specialized task, the number of required features is much smaller than the total model capacity ($K_{\text{task}} \le r \ll d_{\text{model}}$). **Ortho-LoRA** replaces exogenous probes by optimizing a low-rank adapter with a joint **Attribution-Conditioned Gram Loss**:
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}}$

This forces task-relevant feature write-directions to become **mutually orthogonal** (${b}_j \perp {b}_k$) and decorrelates activations directly in the forward pass.

```
                  ┌──────────────────────────────────────────────┐
                  │          The Ortho-LoRA Forward Path         │
                  └──────────────────────────────────────────────┘
                                         x (Input Tokens)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
             W₀ (Frozen Base)                         ΔW = (α/r) · B · A
                   │                                           │
                   │                              ┌────────────┴────────────┐
                   │                              │  a_k^T · x = z_k (Act)  │
                   │                              │  L_sparse Penalty       │
                   │                              │  b_k: Orthogonal Writes │
                   │                              │  L_ortho Gram Penalty   │
                   │                              └────────────┬────────────┘
                   │                                           │
                   ▼                                           ▼
                   h_base ─────────────── + ─────────────── Δh_orthogonal
                                          │
                                          ▼
                                h_adapted (Monosemantic)
```

---

## 📊 Key Findings

1. **Collapse of Superposition:** Ortho-LoRA achieves an **$8.1\times$ to $10.3\times$ reduction in geometric cross-talk** ($C_{gram} = 0.719 \to 0.070$) and increases effective subspace capacity utilization from $57.5\%$ to **$87.7\%$** ($d_{\text{eff}} = 28.05 / 32$).
2. **Causal Steering Selectivity:** Orthogonal channels act as independent causal dials. On counterfactual steering, Ortho-LoRA achieves a **Peak Steering Selectivity Index (SSI) of 3.837**—a **$+41.6\%$ improvement** over Standard LoRA and **$13.2\times$ over Contrastive Activation Addition (CAA)**, while reducing collateral vocabulary drift by $16.3\times$.
3. **Efficiency vs. Top-$K$ SAEs:** Compared to a Top-$K$ SAE ($k=32, 4\times$ expansion):
   - **$112.1\times$ parameter reduction** ($57\text{K}$ vs. $6.4\text{M}$ parameters per layer).
   - **Zero inference latency overhead** when merged into base weights ($\Delta W = BA$).
   - Avoids the severe task loss degradation of SAE activation replacement ($0.658$ vs. $14.944$).
4. **Phase Transition Boundaries:** Validated the Johnson-Lindenstrauss threshold: rank $r \ge 8$ and $\lambda_{\text{ortho}} \ge 1.0$ unpacks geometry cleanly. Restricting capacity to $r=4$ induces capacity-starvation collapse.
5. **Architectural & Domain Universality:** Replicated across Gated SwiGLU (`Qwen-2.5-0.5B`) and Dense GELU (`Pythia-410M`) across affective classification (`tweet_eval`) and safety continuation (`real-toxicity-prompts`).

---

## 🗂 Repository Structure

```plaintext
├── paper/                                  # Complete LaTeX publication source
│   ├── main.tex                            # Two-column conference format paper
│   ├── references.bib                      # BibTeX references
│   └── figures/                            # Vector PDFs and PNGs for publication
│       ├── fig_4_3_sae_training_and_geometry.pdf
│       ├── fig_5_1_geometric_unpacking_and_spaces.pdf
│       ├── fig_5_2_complete_causal_steering.pdf
│       ├── fig_5_3_sae_vs_ortho.pdf
│       ├── fig_6_1_capacity_phase_transition.pdf
│       ├── fig_6_2_layer_ablation.pdf
│       └── fig_6_3_cross_generalization.pdf
├── experiments/                            # Reproducible experiment scripts
│   ├── exp_4_3_train_sae_baseline.py       # Section 4.3: Top-K SAE baseline training
│   ├── exp_5_1_geometric_unpacking.py      # Section 5.1: Weight & activation unpacking
│   ├── exp_5_2_causal_steering.py          # Section 5.2: Counterfactual causal steering & SSI
│   ├── exp_5_3_sae_vs_ortho_benchmark.py   # Section 5.3: Head-to-head comparison
│   ├── exp_6_1_phase_transition.py         # Section 6.1: Rank vs. Lambda grid sweep
│   ├── exp_6_2_layer_ablation.py           # Section 6.2: Early vs. Mid vs. Late layer depth
│   └── exp_6_3_cross_generalization.py     # Section 6.3: Cross-model & cross-task benchmark
├── results/                                # Raw exported benchmark tables (.csv)
│   ├── table_4_3_sae_vs_lora_capacity.csv
│   ├── table_5_1_geometric_unpacking.csv
│   ├── table_5_2_complete_causal_benchmark.csv
│   ├── table_5_3_sae_vs_ortho_benchmark.csv
│   ├── table_6_1_phase_transition_grid.csv
│   ├── table_6_2_layer_depth_benchmark.csv
│   └── table_6_3_cross_generalization_benchmark.csv
├── requirements.txt                        # Minimal dependency specification
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup
All experiments are strictly optimized to execute within a **single 16GB NVIDIA T4 GPU budget** (utilizing $<7\text{ GB}$ peak VRAM under FP16/BF16 mixed precision).

```bash
git clone https://github.com/unfoldinglabs/ortho-lora.git
cd ortho-lora

# Create conda or venv environment
conda create -n ortho-lora python=3.10 -y
conda activate ortho-lora

# Install PyTorch and required dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Dependencies (`requirements.txt`)
```plaintext
torch>=2.1.0
transformers>=4.38.0
datasets>=2.16.0
accelerate>=0.26.0
peft>=0.8.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
```

---

## 🧪 Reproducing the Benchmark Suite

Each script is self-contained, automatically downloads the requisite Hugging Face models and datasets, executes training and evaluation, and logs publication-ready CSVs and vector figures.

### Experiment 4.3: Train Baseline Top-$K$ SAE
Fits an exogenous Top-$K$ SAE ($k=32$, $4\times$ expansion) on Layer 12 residual activations:
```bash
python experiments/exp_4_3_train_sae_baseline.py
```

### Experiment 5.1: Geometric Superposition Collapse
Compares Gram coherence ($C_gram$), Feature Participation Ratio (FPR), and effective dimension ($d_{eff}$):
```bash
python experiments/exp_5_1_geometric_unpacking.py
```

### Experiment 5.2: Causal Activation Steering
Executes counterfactual activation additions across $\gamma \in [-6, +6]$ and computes Steering Selectivity Index (SSI):
```bash
python experiments/exp_5_2_causal_steering.py
```

### Experiment 5.3: Head-to-Head Comparison (Ortho-LoRA vs. Top-$K$ SAE)
Benchmarks parameter overhead, task loss degradation, and token latency:
```bash
python experiments/exp_5_3_sae_vs_ortho_benchmark.py
```

### Experiment 6.1: Capacity & Phase Transition Grid
Sweeps the $4 \times 4$ grid over $r \in \{4, 8, 16, 32\}$ and $\lambda_{ortho} \in [0.0, 10.0]$:
```bash
python experiments/exp_6_1_phase_transition.py
```

### Experiment 6.2: Architectural Depth Ablation
Evaluates early (25%), mid (50%), and late (75%) layer interventions:
```bash
python experiments/exp_6_2_layer_ablation.py
```

### Experiment 6.3: Cross-Model & Cross-Domain Generalization
Runs the full evaluation across `Qwen-2.5-0.5B` and `Pythia-410M` on sentiment and toxicity tasks:
```bash
python experiments/exp_6_3_cross_generalization.py
```

---

## 📖 Citation

If you use this codebase, methodology, or experimental findings in your research, please cite:

```bibtex
@article{nassimberrada2026ortholora,
  title   = {Endogenous Subspace Orthogonalization: Targeted Monosemanticity via Low-Rank Adaptation},
  author  = {Nassim berrada},
  journal = {Preprint},
  year    = {2026}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
