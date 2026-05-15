#!/bin/bash
cd ~/repositories/long-context-retrieval
source .venv/bin/activate

BASE_PATH="./data/processed/polish_final_cluster_reparsed_waterfall/compare"
EMBEDDER_MODEL="BAAI/bge-m3"
QUERIES_PATH="$BASE_PATH/contextual_queries.jsonl"

# 1. Base (no waterfall)
echo "Evaluating base (no waterfall) with bge-m3"
python3 lcr/cli/eval.py \
    --chunks-path "$BASE_PATH/augmented_chunks_obligation_to_defend.jsonl" \
    --model-path "$EMBEDDER_MODEL" \
    --results-jsonl-path "$BASE_PATH/results/bge-m3_base_results.jsonl" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "flag" \
    --encoding-type "dense" \
    --batch-size 32

# 2. Waterfall append
echo "Evaluating waterfall append with bge-m3"
python3 lcr/cli/eval.py \
    --chunks-path "$BASE_PATH/augmented_chunks_obligation_to_defend_waterfall_append.jsonl" \
    --model-path "$EMBEDDER_MODEL" \
    --results-jsonl-path "$BASE_PATH/results/bge-m3_waterfall_append_results.jsonl" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "flag" \
    --encoding-type "dense" \
    --batch-size 32

# 3. Waterfall consolidate
echo "Evaluating waterfall consolidate with bge-m3"
python3 lcr/cli/eval.py \
    --chunks-path "$BASE_PATH/augmented_chunks_obligation_to_defend_waterfall_consolidate.jsonl" \
    --model-path "$EMBEDDER_MODEL" \
    --results-jsonl-path "$BASE_PATH/results/bge-m3_waterfall_consolidate_results.jsonl" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "flag" \
    --encoding-type "dense" \
    --batch-size 32
