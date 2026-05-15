#!/bin/bash


# SAVE_DIR="./data/processed/danish_dataset_7_no_chain/merged_q_r45_filter"
# CHUNKS_PATH="./data/processed/danish_dataset_7_no_chain/merged/chunks.jsonl"
# QUERIES_PATH="./data/processed/danish_dataset_7_no_chain/merged_r45/queries.jsonl"
RESULTS_JSONL_PATH="$1"
CHUNKS_PATH="$2"
QUERIES_PATH="$3"
ENCODING_TYPE="${4:-dense}"
EMBEDDER_MODEL="${5:-BAAI/bge-m3}"  # or leave blank for default
python3 lcr/cli/eval.py \
    --model-path "$EMBEDDER_MODEL" \
    --chunks-path "$CHUNKS_PATH" \
    --results-jsonl-path "$RESULTS_JSONL_PATH" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "flag" \
    --encoding-type "$ENCODING_TYPE"