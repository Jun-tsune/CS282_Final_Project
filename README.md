# CS282-Fall2025 Final Project

This is the repository for the final project of CS282-Fall2025.

## Project Overview

This project studies how compressive memory affects In-Context Learning (ICL) in long-sequence Transformers. Using synthetic tasks, we compare standard and Compressive Transformers across context lengths and memory ratios. We expect compressive memory to enhance long-range ICL while showing diminishing gains as compression increases.

The project is based on the following repositories:
* [What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://github.com/dtsip/in-context-learning)
* [Compressive Transformers for Long-Range Sequence Modelling](https://github.com/lucidrains/compressive-transformer-pytorch)

## File Structure

```
CS282_Final_Project/
├── src/                    # Source code
│   ├── models/            # Model architectures (Transformer, CompressiveTransformer)
│   ├── config/            # Configuration files
│   │   ├── config_model/  # Model configurations
│   │   └── config_train/  # Training configurations
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   ├── eval.py            # Evaluation script
│   ├── eval.ipynb         # Evaluation notebook
│   └── tasks.py           # ICL task definitions
├── scripts/                # Bash scripts
│   ├── train.sh          # Training launcher
│   ├── eval.sh           # Evaluation launcher
│   └── train_batch.sh    # Batch training script
├── outputs/               # Model checkpoints and results (not in git)
├── logs/                  # Training and evaluation logs
├── notebooks/             # Jupyter notebooks for analysis
├── environment.yml        # Conda environment file
└── README.md             # This file
```

**Note:** The `outputs/` directory contains model checkpoints and training results. These files are large (400MB+) and are excluded from git via `.gitignore`. To reproduce results, you need to train the models yourself (see Training section below).

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Jun-tsune/CS282_Final_Project.git
cd CS282_Final_Project
```

### 2. Install dependencies

```bash
conda env create -f environment.yml
conda activate env_282PJ
```

### 3. Verify installation

The project uses PyTorch, wandb, and other dependencies. Make sure you have CUDA available if training on GPU.

## Usage

### Training

#### Basic Training

Train a model using configuration files:

```bash
bash scripts/train.sh <MODEL_YAML> <TRAIN_YAML>
```

**Example:**
```bash
# Train a Compressive Transformer with sequence length 256
bash scripts/train.sh config_model_comp config_train_seq256

# Train a standard Transformer with sequence length 512
bash scripts/train.sh config_model_trans config_train_seq512
```

#### Resume Training

To resume from a checkpoint:

```bash
bash scripts/train.sh <MODEL_YAML> <TRAIN_YAML> \
    --resume_output_id <RUN_ID> \
    --wandb_id <WANDB_ID>
```

**Example:**
```bash
bash scripts/train.sh config_model_comp config_train_seq256 \
    --resume_output_id comp_seq256_20251128_143606 \
    --wandb_id iq2k7biy
```

#### Direct Python Training

You can also run training directly with Python:

```bash
python src/train.py \
    model_yaml=config_model_comp \
    train_yaml=config_train_seq256 \
    training.resume_id=<run_id> \
    wandb.name=<wandb_name> \
    training.device=cuda
```

### Evaluation

#### Evaluate a Single Run

```bash
bash scripts/eval.sh <RUN_ID or OUTPUT_DIR>
```

**Examples:**
```bash
# Using run ID
bash scripts/eval.sh comp_seq256_20251128_143606

# Using full path
bash scripts/eval.sh outputs/id_comp_seq256_20251128_143606

# Evaluate all strategies (slower but more comprehensive)
bash scripts/eval.sh outputs/id_comp_seq256_20251128_143606 --all-strategies
```

#### Direct Python Evaluation

```bash
# Standard strategy only (faster)
python src/eval.py outputs/id_<run_id>

# All strategies
python src/eval.py outputs/id_<run_id> --all-strategies
```

#### Evaluation Notebook

For interactive evaluation and analysis, use the Jupyter notebook:

```bash
jupyter notebook src/eval.ipynb
```

### Configuration Files

Configuration files are located in `src/config/`:

- **Model configs** (`config_model/`): Define model architecture
  - `config_model_trans.yaml`: Standard Transformer
  - `config_model_comp.yaml`: Compressive Transformer
  - `config_model_ratio*.yaml`: Different compression ratios

- **Training configs** (`config_train/`): Define training parameters
  - `config_train_seq*.yaml`: Different sequence lengths
  - `config_train_cmem.yaml`: Compressive memory settings

## Model Architectures

### Standard Transformer
- Standard attention mechanism
- Full context window

### Compressive Transformer
- Compressive memory mechanism
- Supports different compression ratios (`cmem_ratio`)
- Includes reconstruction loss for memory compression

## Evaluation Strategies

The evaluation framework supports multiple strategies:
- `standard`: Standard evaluation
- `random_quadrants`: Random quadrant sampling
- `orthogonal_train_test`: Orthogonal train/test split
- And more (see `src/eval.py` for full list)

By default, only the `standard` strategy is evaluated for faster results. Use `--all-strategies` flag to evaluate all strategies.

## Results

Model checkpoints and evaluation results are saved in `outputs/id_<run_id>/`:
- `model_*.pt`: Model checkpoints at different training steps
- `state.pt`: Latest model state
- `config.yaml`: Configuration used for training
- `metrics.json`: Evaluation metrics (generated after running eval.py)
- `wandb/`: Weights & Biases logs

## Notes

- Model checkpoints are **not** included in the git repository due to size constraints
- To reproduce results, you need to train models from scratch
- Training logs are saved in `logs/` directory
- WandB is used for experiment tracking (configure in `src/config/standard.yaml`)

## Contributing

This is a course project repository. For questions or issues, please contact the repository maintainers.