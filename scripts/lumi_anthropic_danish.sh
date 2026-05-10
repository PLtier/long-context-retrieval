#!/bin/bash
cd ~/repositories/long-context-retrieval
source .venv/bin/activate


RAW_BASE="./data/raw/danish_final_cluster"
PROCESSED_BASE="./data/processed/danish_final_cluster_r10_ablation"
CONTEXTUALISATION_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

VLLM_BASE_URL=http://localhost:8000/v1 bash scripts/anthropic_contextualise.sh "$RAW_BASE" "$PROCESSED_BASE" "$CONTEXTUALISATION_MODEL"

# merge contextualised_chunks.jsonl into merged/contextualised_chunks.jsonl
MERGED_DIR="$PROCESSED_BASE/merged"
bash scripts/merge.sh "$PROCESSED_BASE" "$MERGED_DIR" contextualised_chunks.jsonl