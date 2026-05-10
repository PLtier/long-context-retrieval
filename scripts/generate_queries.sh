#!/bin/bash

# Iterate through all clusters and subfolders in datasets and run the two commands for each dataset directory
# RAW_BASE=./data/raw/polish_dataset_7
# PROCESSED_BASE=./data/processed/polish_dataset_7_no_chain


RAW_BASE="$1"
PROCESSED_BASE="$2"
PROMPT_TEMPLATE="$3"

if [ -z "$RAW_BASE" ] || [ -z "$PROCESSED_BASE" ] || [ -z "$PROMPT_TEMPLATE" ]; then
    echo "Usage: $0 /path/to/raw_base /path/to/processed_base /path/to/prompt_template"
    exit 1
fi

for cluster in "$RAW_BASE"/cluster_*; do
    for dataset in "$cluster"/*; do
        if [ -d "$dataset" ] && [ -f "$dataset/chunks.jsonl" ]; then
            dataset_name=$(basename "$dataset")
            cluster_name=$(basename "$cluster")
            echo "Processing $cluster_name/$dataset_name"
            python lcr/cli/generate_queries.py generate-queries \
                --prompt-template "$PROMPT_TEMPLATE" \
                --documents-base-dir "$dataset" \
                --datasets chunks \
                --save-path "$PROCESSED_BASE/$cluster_name/$dataset_name" \
                --llm "openai/gpt-oss-120b" \
                --provider "vllm" \
                --impl-context-col "implicit_context_chunks" \
                --context-col "explicit_context_chunks" \
                --no-chain-context \
                --no-update-queries \
                --start-from-checkpoint 
            python lcr/visualisation/visualise_jsonl.py "$PROCESSED_BASE/$cluster_name/$dataset_name/queries.jsonl" --preset queries
        fi
    done
done
