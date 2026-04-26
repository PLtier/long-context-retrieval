#!/bin/bash

# Iterate through all clusters and subfolders in danish_dataset_7 and run the two commands for each dataset directory with queries.jsonl
PROCESSED_BASE=./data/processed/polish_dataset_7_no_chain

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
                --no-start-from-checkpoint \
                --llm "ge" \
                --provider "openrouter"
            python lcr/visualisation/visualise_jsonl.py "$dataset/assurance_results.jsonl" --preset assurance_results
        fi
    done
done