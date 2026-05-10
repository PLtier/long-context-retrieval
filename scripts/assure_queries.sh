#!/bin/bash

# Iterate through all clusters and subfolders in danish_dataset_7 and run the two commands for each dataset directory with queries.jsonl
# PROCESSED_BASE=./data/processed/polish_dataset_7_no_chain
PROCESSED_BASE="$1"
PROMPT_TEMPLATE="$2"

if [ -z "$PROCESSED_BASE" ] || [ -z "$PROMPT_TEMPLATE" ]; then
    echo "Usage: $0 /path/to/processed_base /path/to/prompt_template"
    exit 1
fi

for cluster in "$PROCESSED_BASE"/cluster_*; do
    for dataset in "$cluster"/*; do
        if [ -d "$dataset" ] && [ -f "$dataset/queries.jsonl" ]; then
            dataset_name=$(basename "$dataset")
            cluster_name=$(basename "$cluster")
            echo "Assuring $cluster_name/$dataset_name"
            python lcr/cli/generate_queries.py assure-queries \
                --queries-base-dir "$dataset" \
                --datasets queries \
                --save-path "$dataset" \
                --start-from-checkpoint \
                --llm "openai/gpt-oss-120b" \
                --provider "vllm" \
                --prompt-template "$PROMPT_TEMPLATE"
            python lcr/visualisation/visualise_jsonl.py "$dataset/assurance_results.jsonl" --preset assurance_results
        fi
    done
done