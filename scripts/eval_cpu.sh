#!/bin/bash


SAVE_DIR="./data/processed/danish_dataset_7_no_chain/cluster_B/erhversfondsloven"
CHUNKS_PATH="./data/processed/danish_dataset_7_no_chain/cluster_B/erhversfondsloven/augmented_chunks.jsonl"
EMBEDDER_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # or leave blank for default
QUERIES_PATH="./data/processed/danish_dataset_7_no_chain/cluster_B/erhversfondsloven/queries.jsonl"
python3 lcr/cli/eval.py \
    --model-path "$EMBEDDER_MODEL" \
    --chunks-path "$CHUNKS_PATH" \
    --save-results-dir "$SAVE_DIR" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \