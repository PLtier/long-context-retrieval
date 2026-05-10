#!/bin/bash

RAW_BASE="$1"
PROCESSED_BASE="$2"
CONTEXTUALISATION_MODEL="$3"

if [ -z "$RAW_BASE" ] || [ -z "$PROCESSED_BASE" ] || [ -z "$CONTEXTUALISATION_MODEL" ]; then
    echo "Usage: $0 /path/to/raw_base /path/to/processed_base /path/to/contextualisation_model"
    exit 1
fi

for cluster in "$RAW_BASE"/cluster_*; do
    for dataset in "$cluster"/*; do
        if [ -d "$dataset" ] && [ -f "$dataset/chunks.jsonl" ]; then
            dataset_name=$(basename "$dataset")
            cluster_name=$(basename "$cluster")
            echo "Processing $cluster_name/$dataset_name"
            python3 lcr/cli/waterfall_contextualise_dataset.py \
                --contextualisation-model "$CONTEXTUALISATION_MODEL" \
                --chunks-path "$dataset/chunks.jsonl" \
                --save-dir "$PROCESSED_BASE/$cluster_name/$dataset_name" \
                --provider "vllm" \
                --max-concurrent 64 \
                --start-from-checkpoint
        fi
    done
done
