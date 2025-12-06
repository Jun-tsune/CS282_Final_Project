#!/bin/bash
#
# Generic evaluation script
# Usage:
#   bash scripts/eval.sh <RUN_ID or OUTPUT_DIR> [--all-strategies]
#   Example: bash scripts/eval.sh compressive_seq256_20251116_211121
#   Or:      bash scripts/eval.sh outputs/id_compressive_seq256_20251116_211121
#   Or:      bash scripts/eval.sh outputs/id_compressive_seq256_20251116_211121 --all-strategies
#
# By default, only evaluates 'standard' strategy for faster evaluation.
# Use --all-strategies to evaluate all 14 strategies.

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Activate conda environment
if command -v conda &> /dev/null; then
    # Try to initialize conda if not already initialized
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        source ~/.bashrc 2>/dev/null || true
        # Try common conda initialization paths
        if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        fi
    fi
    conda activate env_282PJ 2>/dev/null || {
        echo "Warning: Could not activate conda environment env_282PJ"
        echo "Make sure conda is initialized and environment exists"
        echo "You may need to run: conda activate env_282PJ manually"
    }
else
    echo "Warning: conda not found, skipping environment activation"
    echo "Make sure Python dependencies are available in current environment"
fi

# Check if run ID or output directory is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a RUN_ID or output directory"
    echo "Usage: bash scripts/eval.sh <RUN_ID or OUTPUT_DIR> [--all-strategies]"
    echo "Example: bash scripts/eval.sh compressive_seq256_20251116_211121"
    echo "Or:      bash scripts/eval.sh outputs/id_compressive_seq256_20251116_211121"
    echo "Or:      bash scripts/eval.sh outputs/id_compressive_seq256_20251116_211121 --all-strategies"
    echo ""
    echo "By default, only evaluates 'standard' strategy for faster evaluation."
    echo "Use --all-strategies to evaluate all 14 strategies."
    exit 1
fi

# Check for --all-strategies flag
EVAL_ALL_STRATEGIES=false
if [ "$2" == "--all-strategies" ]; then
    EVAL_ALL_STRATEGIES=true
fi

# Determine output directory
if [[ "$1" == outputs/id_* ]] || [[ "$1" == /* ]] || [[ "$1" == ./* ]]; then
    # Full path provided (absolute, relative, or starting with outputs/id_)
    if [[ "$1" == outputs/id_* ]]; then
        OUTPUT_DIR="$1"
    else
        OUTPUT_DIR="$1"
    fi
    RUN_ID=$(basename "$OUTPUT_DIR" | sed 's/^id_//')
else
    # Just run ID provided
    RUN_ID="$1"
    OUTPUT_DIR="outputs/id_${RUN_ID}"
fi

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: ${OUTPUT_DIR}"
    echo "Please check the RUN_ID or output directory path"
    exit 1
fi

# Extract model info from RUN_ID for logging purposes
# Try to extract model type and sequence length from RUN_ID
# Format examples: compressive_seq256_20251116_211121, transformer_seq128_20251116_211121
MODEL_INFO=""
if [[ "$RUN_ID" =~ ^([^_]+)_seq([0-9]+)_ ]]; then
    MODEL_TYPE="${BASH_REMATCH[1]}"
    SEQ_LEN="${BASH_REMATCH[2]}"
    MODEL_INFO="${MODEL_TYPE} (seq=${SEQ_LEN})"
else
    # Use first part of RUN_ID as model identifier
    MODEL_INFO=$(echo "$RUN_ID" | cut -d'_' -f1)
fi

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Create log filename based on RUN_ID: eval_{run_id_base}_{timestamp}.out
# Extract base name from RUN_ID (remove timestamp if present)
RUN_ID_BASE=$(echo "$RUN_ID" | sed -E 's/_[0-9]{8}_[0-9]{6}$//' | sed -E 's/_[0-9]+$//')
# Fallback to full RUN_ID if extraction fails
if [ -z "$RUN_ID_BASE" ] || [ "$RUN_ID_BASE" == "$RUN_ID" ]; then
    RUN_ID_BASE="$RUN_ID"
fi
LOG_FILE="${LOG_DIR}/eval_${RUN_ID_BASE}_${TIMESTAMP}.out"
exec > >(tee "$LOG_FILE") 2>&1

echo "========================================="
echo "Evaluating Model"
if [ -n "$MODEL_INFO" ]; then
    echo "Model: ${MODEL_INFO}"
fi
echo "Run ID: ${RUN_ID}"
echo "Output directory: ${OUTPUT_DIR}"
if [ "$EVAL_ALL_STRATEGIES" = true ]; then
    echo "Mode: All strategies (14 strategies)"
    EVAL_ARGS="${OUTPUT_DIR} --all-strategies"
else
    echo "Mode: Standard strategy only (faster)"
    EVAL_ARGS="${OUTPUT_DIR}"
fi
echo "Started at: $(date)"
echo "Log file: ${LOG_FILE}"
echo "========================================="
echo ""

# Run evaluation
python src/eval.py ${EVAL_ARGS}

EVAL_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Evaluation completed at: $(date)"
echo "Exit code: ${EVAL_EXIT_CODE}"
echo "========================================="

exit ${EVAL_EXIT_CODE}

