#!/bin/bash

# Usage: ./iter_queries.sh /path/to/dataset_dir /path/to/output_dir
# Example: ./iter_queries.sh ./data/processed/polish_dataset_7_no_chain/cluster_A/dataset_1 ./data/processed/polish_dataset_7_no_chain_is_questions/cluster_A/dataset_1

DATASET_DIR="$1"
OUTPUT_DIR="$2"

if [ -z "$DATASET_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 /path/to/dataset_dir /path/to/output_dir"
    exit 1
fi

if [ -d "$DATASET_DIR" ] && [ -f "$DATASET_DIR/queries.jsonl" ]; then
    echo "Processing $DATASET_DIR -> $OUTPUT_DIR"
    python lcr/cli/generate_queries.py generate-queries \
        --documents-base-dir "$DATASET_DIR" \
        --datasets queries \
        --save-path "$OUTPUT_DIR" \
        --no-start-from-checkpoint \
        --llm "OpenAI/gpt-oss-120B" \
        --provider "together" \
        --impl-context-col "impl_context_chunks" \
        --context-col "context_chunks" \
        --no-chain-context
    python lcr/visualisation/visualise_jsonl.py "$OUTPUT_DIR/queries.jsonl" --preset queries
else
    echo "Dataset directory or queries.jsonl not found: $DATASET_DIR"
    exit 2
fi
