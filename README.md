# Elasticity-Induced Reversion in Fine-Tuned Traffic Language Models

This repository contains a complete implementation of the paper "Elasticity Induced Reversion in Fine-Tuned Traffic Language Models under Perturbations" (ACM).

## Overview

This project implements an **elasticity-guided framework** for discovering minimal perturbation sets that induce **rebound behavior** in fine-tuned traffic language models. The framework reveals fundamental robustness limitations in LLM-based encrypted traffic detectors.

### Key Contributions

1. **Elastic Rebound Theory**: First systematic study of rebound in traffic-oriented LLMs under perturbations
2. **Elasticity Energy Metric**: Computable energy \($E(u) = \frac{1}{2}\tilde{d}^T g_{proxy}(u))^2$\)for zero-query perturbation ranking  
3. **Dual-Layer Evolution**: Submodular coverage + constrained GA for efficient perturbation discovery
4. **Protocol-Compliant Perturbations**: Genome encoding for realistic packet-level manipulations

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.3+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/HHLinlife/NetelasticLLM.git
cd NetelasticLLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
NetelasticLLM/
├── config/                     # Configuration files
│   ├── model_config.yaml      # Model architecture settings
│   ├── training_config.yaml   # Training parameters
│   └── perturbation_config.yaml  # Perturbation constraints
│
├── data/                       # Data processing modules
│   ├── datasets/              # Dataset implementations
│   │   ├── base_dataset.py   # Abstract base class
│   │   ├── ustc_tfc2016.py  # USTC-TFC2016 dataset
│   │   ├── cic_darknet2020.py  # CIC-Darknet2020 dataset
│   │   └── cesnet_tls22.py  # CESNET-TLS22 dataset
│   └── preprocessing/         # Flow parsing utilities
│       ├── flow_parser.py    # PCAP to flow conversion
│       └── feature_extractor.py  # Feature extraction
│
├── models/                     # Model implementations
│   ├── backbone/              # Transformer backbone
│   │   ├── transformer_backbone.py  # LLaMA-style decoder
│   │   └── pretrained_llm.py  # Pretrained model wrapper
│   ├── traffic_classifier/    # Traffic classification
│   │   ├── traffic_llm.py    # Complete TrafficLLM model
│   │   └── fine_tuning_strategies.py  # LoRA, Prefix tuning
│   └── surrogate/             # Surrogate model
│       └── surrogate_model.py  # Lightweight distilled model
│
├── perturbation/               # Perturbation framework
│   ├── genome_encoding.py     # Variable-length genome: g=(d,l,δ)
│   ├── mutation_operators.py  # Direction, length, timing ops
│   ├── elasticity_energy.py   # E(u) computation
│   └── constraints.py         # Protocol & physical constraints
│
├── search/                     # Evolutionary search
│   ├── submodular_selection.py  # Layer I: Coverage
│   ├── genetic_algorithm.py   # Layer II: GA refinement
│   └── dual_layer_evolution.py  # Complete Algorithm 1
│
├── distillation/               # Surrogate distillation
│   ├── one_shot_distillation.py  # One-shot query protocol
│   └── active_query_selection.py  # Active learning
│
├── evaluation/                 # Evaluation metrics
│   ├── metrics.py             # KL divergence, accuracy
│   └── rebound_analyzer.py    # Rebound analysis
│
├── experiments/                # Experiment scripts
│   ├── run_experiment.py      # Main experiment runner
│   └── ablation_study.py      # Ablation studies
│
├── utils/                      # Utilities
│   ├── logger.py              # Logging configuration
│   └── visualization.py       # Result visualization
│
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Prepare Data

Place your encrypted traffic datasets in the `data/` directory:

```bash
data/
├── USTC-TFC2016/
│   ├── benign/
│   └── malware/
├── CIC-Darknet2020/
└── CESNET-TLS22/
```

### 2. Train Pretrained Model

```bash
python main.py --mode pretrain \
    --dataset ustc_tfc2016 \
    --model_size 8B \
    --epochs 100
```

### 3. Fine-tune for Fine-grained Classification

```bash
python main.py --mode finetune \
    --dataset ustc_tfc2016 \
    --pretrained_checkpoint checkpoints/pretrained_model.pt \
    --finetuning_method lora \
    --alignment_size 5000
```

### 4. Distill Surrogate Model

```bash
python main.py --mode distill \
    --finetuned_checkpoint checkpoints/finetuned_model.pt \
    --query_budget 1500
```

### 5. Discover Rebound Perturbations

```bash
python main.py --mode search \
    --pretrained_checkpoint checkpoints/pretrained_model.pt \
    --surrogate_checkpoint checkpoints/surrogate_model.pt \
    --num_prototypes 50 \
    --max_generations 100
```

### 6. Evaluate Rebound Behavior

```bash
python main.py --mode evaluate \
    --perturbation_set results/perturbations.pkl \
    --compute_kl_divergence \
    --analyze_rebound
```

## Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
backbone:
  architecture: "llama"  # llama, qwen, deepseek
  selected_size: "small"  # small (8B), medium (70B), large (405B)
  
traffic_classifier:
  num_coarse_classes: 2   # benign vs malicious
  num_fine_classes: 20    # specific categories
  
fine_tuning:
  methods:
    lora:
      rank: 8
      alpha: 32
    prefix:
      prefix_length: 128
```

### Perturbation Configuration (`config/perturbation_config.yaml`)

```yaml
genome:
  direction_values: [-1, 1]
  length_bounds: [64, 1500]  # bytes
  timing_bounds: [0, 1000]   # milliseconds
  
constraints:
  max_total_bytes: 1048576    # 1 MB
  max_total_duration: 60000   # 60 seconds
  
evolution:
  num_prototypes: 50
  mu: 30  # elite size
  lambda: 60  # offspring size
  max_generations: 100
```



## Experimental Results

### Rebound Under Different Alignment Depths

| Dataset | θ₁⁽¹⁾ | θ₁⁽²⁾ | θ₁⁽³⁾ | θ₁⁽⁴⁾ |
|---------|--------|--------|--------|--------|
| USTC-TFC2016 | -5.2% | -12.8% | -18.4% | -20.6% |
| CIC-Darknet2020 | -6.1% | -14.3% | -19.7% | -22.2% |
| CESNET-TLS22 | -5.8% | -13.9% | -20.1% | -22.3% |
| Private datasets | -7.3% | -16.2% | -22.8% | -24.8% |

### Effect of Pretraining Scale

Larger pretraining corpora consistently amplify rebound strength:
- 0.07T tokens: ΔAcc = -15.9%
- 0.56T tokens: ΔAcc = -23.6% (+48% stronger rebound)

### Effect of Parameter Scale

Rebound increases monotonically with model size:
- 8B parameters: KL ↓ 0.106 → 0.083 (-21.7%)
- 70B parameters: KL ↓ 0.106 → 0.059 (-44.3%)
