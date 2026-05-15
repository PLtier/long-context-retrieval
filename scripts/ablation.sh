#!/bin/bash
cd ~/repositories/long-context-retrieval
source .venv/bin/activate

PROCESSED_BASE="$1"


MERGED_DIR="$PROCESSED_BASE/merged"

# This file runs 3 configurations:
# 1. Weak augmentation and weak dense retrieval
# 2. Strong augmentation and weak dense retrieval
# 3. Weak augmentation and strong dense retrieval
# Best one is located in other repo.
WEAK_EMBEDDER_MODEL="intfloat/multilingual-e5-small" # by default, uses 512 window.
STRONG_EMBEDDER_MODEL="BAAI/bge-m3" # uses 8192 window
WEAK_AUGMENTATION_FILE="$MERGED_DIR/augmented_chunks30B.jsonl"
STRONG_AUGMENTATION_FILE="$MERGED_DIR/augmented_chunks235B.jsonl"
QUERIES_PATH="$MERGED_DIR/contextual_queries.jsonl"


# 1. Weak augmentation and weak dense retrieval
echo "Weak augmentation and weak dense retrieval"
python3 lcr/cli/eval.py \
    --chunks-path "$WEAK_AUGMENTATION_FILE" \
    --model-path "$WEAK_EMBEDDER_MODEL" \
    --results-jsonl-path "$MERGED_DIR/qwen3_30B_me5_results.jsonl" \
    --queries-path "$QUERIES_PATH" \
    --use-prefix \
    --encoder "sentence" \

# 2. Strong augmentation and weak dense retrieval
echo "Strong augmentation and weak dense retrieval"
python3 lcr/cli/eval.py \
    --chunks-path "$STRONG_AUGMENTATION_FILE" \
    --model-path "$WEAK_EMBEDDER_MODEL" \
    --results-jsonl-path "$MERGED_DIR/qwen3_235B_me5_results.jsonl" \
    --queries-path "$QUERIES_PATH" \
    --use-prefix \
    --encoder "sentence" \


# 3. Weak augmentation and strong dense retrieval
echo "Weak augmentation and strong dense retrieval"
python3 lcr/cli/eval.py \
    --chunks-path "$WEAK_AUGMENTATION_FILE" \
    --model-path "$STRONG_EMBEDDER_MODEL" \
    --results-jsonl-path "$MERGED_DIR/qwen3_30B_bge-m3_results.jsonl" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "flag" \
    --encoding-type "dense" \
    --batch-size 16

# 4. Strong augmentation and strong dense retrieval
echo "Strong augmentation and strong dense retrieval"
python3 lcr/cli/eval.py \
    --chunks-path "$STRONG_AUGMENTATION_FILE" \
    --model-path "$STRONG_EMBEDDER_MODEL" \
    --results-jsonl-path "$MERGED_DIR/qwen3_235B_bge-m3_results.jsonl" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "flag" \
    --encoding-type "dense" \
    --batch-size 32