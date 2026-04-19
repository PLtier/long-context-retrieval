#!/bin/bash

# Iterate through all clusters and subfolders in danish_dataset_7 and run the two commands for each dataset directory
RAW_BASE=./data/raw/danish_dataset_7
PROCESSED_BASE=./data/processed/danish_dataset_7_no_chain

for cluster in "$RAW_BASE"/cluster_*; do
    for dataset in "$cluster"/*; do
        if [ -d "$dataset" ] && [ -f "$dataset/chunks.jsonl" ]; then
            dataset_name=$(basename "$dataset")
            cluster_name=$(basename "$cluster")
            echo "Processing $cluster_name/$dataset_name"
            python lcr/generate_queries.py generate-queries \
                --documents-base-dir "$dataset" \
                --datasets chunks \
                --save-path "$PROCESSED_BASE/$cluster_name/$dataset_name" \
                --no-start-from-checkpoint \
                --llm "OpenAI/gpt-oss-120B" \
                --provider "together" \
                --impl-context-col "implicit_context_chunks" \
                --context-col "explicit_context_chunks" \
                --no-chain-context
            python lcr/visualisation/visualise_jsonl.py "$PROCESSED_BASE/$cluster_name/$dataset_name/queries.jsonl" --preset queries
        fi
    done
done
