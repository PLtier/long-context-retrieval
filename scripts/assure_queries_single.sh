#!/bin/bash

# Usage: ./assure_queries_single.sh /path/to/dataset_dir
# Example: ./assure_queries_single.sh ./data/processed/polish_dataset_7_no_chain/cluster_A/dataset_1

DATASET_DIR="$1"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 /path/to/dataset_dir"
    exit 1
fi

if [ -d "$DATASET_DIR" ] && [ -f "$DATASET_DIR/queries.jsonl" ]; then
    echo "Assuring $DATASET_DIR"
    python lcr/cli/generate_queries.py assure-queries \
        --queries-base-dir "$DATASET_DIR" \
        --datasets queries \
        --save-path "$DATASET_DIR" \
        --no-start-from-checkpoint \
        --llm "OpenAI/gpt-oss-120B" \
        --provider "together"
    python lcr/visualisation/visualise_jsonl.py "$DATASET_DIR/assurance_results.jsonl" --preset assurance_results
else
    echo "Dataset directory or queries.jsonl not found: $DATASET_DIR"
    exit 2
fi
