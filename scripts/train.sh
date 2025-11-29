#!/bin/bash
#
# Generic training script
# Usage:
#   bash scripts/train.sh MODEL_YAML TRAIN_YAML [RUN_ID]
#   bash scripts/train.sh config_model_comp_seq256 config_train_seq256
#   bash scripts/train.sh config_model_comp_seq256 config_train_seq256 my_custom_run_id
#
#   If RUN_ID is not provided, a new one will be generated based on model_yaml and train_yaml

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: MODEL_YAML and TRAIN_YAML are required"
    echo "Usage: bash scripts/train.sh MODEL_YAML TRAIN_YAML [RUN_ID]"
    echo "Example: bash scripts/train.sh config_model_comp_seq256 config_train_seq256"
    exit 1
fi

MODEL_YAML="$1"
TRAIN_YAML="$2"
RUN_ID="$3"

# # Activate conda environment
# if command -v conda &> /dev/null; then
#     # Try to initialize conda if not already initialized
#     if [ -z "$CONDA_DEFAULT_ENV" ]; then
#         source ~/.bashrc 2>/dev/null || true
#         # Try common conda initialization paths
#         if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#             source "$HOME/anaconda3/etc/profile.d/conda.sh"
#         elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#             source "$HOME/miniconda3/etc/profile.d/conda.sh"
#         fi
#     fi
#     conda activate env_282PJ 2>/dev/null || {
#         echo "Warning: Could not activate conda environment env_282PJ"
#         echo "Make sure conda is initialized and environment exists"
#         echo "You may need to run: conda activate env_282PJ manually"
#     }
# else
#     echo "Warning: conda not found, skipping environment activation"
#     echo "Make sure Python dependencies are available in current environment"
# fi

# Extract model type and sequence length from yaml names for naming
# Remove 'config_model_' or 'config_train_' prefix
MODEL_BASE=$(echo "$MODEL_YAML" | sed 's/^config_model_//')
TRAIN_BASE=$(echo "$TRAIN_YAML" | sed 's/^config_train_//')

# Generate run ID if not provided
if [ -z "$RUN_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    # Use model_base as the base for run_id (e.g., comp_seq256 -> comp_seq256_timestamp)
    RUN_ID="${MODEL_BASE}_${TRAIN_BASE}_${TIMESTAMP}"
fi

# Generate log file name based on model and train yaml
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Create log filename: train_{model_base}_{timestamp}.out
LOG_FILE="${LOG_DIR}/train_${MODEL_BASE}_${TIMESTAMP}.out"
exec > >(tee "$LOG_FILE") 2>&1

echo "========================================="
echo "Training"
echo "Model YAML: ${MODEL_YAML}"
echo "Train YAML: ${TRAIN_YAML}"
echo "Run ID: ${RUN_ID}"
echo "Started at: $(date)"
echo "Log file: ${LOG_FILE}"
echo "========================================="
echo ""
echo "Device Configuration:"
echo "  Requested device: cuda"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | sed 's/^/    /'
else
    echo "  nvidia-smi not available (may not have GPU access)"
fi
echo ""

# Run training
python src/train.py \
    test_run=False \
    model_yaml="${MODEL_YAML}" \
    train_yaml="${TRAIN_YAML}" \
    training.resume_id="${RUN_ID}" \
    wandb.name="${RUN_ID}" \
    training.device='cuda'

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Training completed at: $(date)"
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "Output directory: outputs/id_${RUN_ID}"
echo "========================================="

exit ${TRAIN_EXIT_CODE}

