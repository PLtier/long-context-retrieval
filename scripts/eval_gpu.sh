#!/bin/bash


# SAVE_DIR="./data/processed/danish_dataset_7_no_chain/merged_q_r45_filter"
# CHUNKS_PATH="./data/processed/danish_dataset_7_no_chain/merged/chunks.jsonl"
EMBEDDER_MODEL="BAAI/bge-m3"  # or leave blank for default
# QUERIES_PATH="./data/processed/danish_dataset_7_no_chain/merged_r45/queries.jsonl"
SAVE_DIR="$1"
CHUNKS_PATH="$2"
QUERIES_PATH="$3"
python3 lcr/cli/eval.py \
    --model-path "$EMBEDDER_MODEL" \
    --chunks-path "$CHUNKS_PATH" \
    --save-results-dir "$SAVE_DIR" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "flag"

python lcr/visualisation/visualise_jsonl.py "$SAVE_DIR/eval_results.jsonl"