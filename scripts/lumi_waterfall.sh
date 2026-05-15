
#!/bin/bash
cd ~/repos/long-context-retrieval
source .venv/bin/activate


RAW_BASE="./data/raw/polish_final_cluster_reparsed"
PROCESSED_BASE="./data/processed/polish_final_cluster_reparsed_waterfall"
CONTEXTUALISATION_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
VLLM_HOST="${VLLM_HOST:-localhost}"

VLLM_BASE_URL=http://${VLLM_HOST}:${VLLM_PORT:-8000}/v1 bash scripts/waterfall_contextualise.sh "$RAW_BASE" "$PROCESSED_BASE" "$CONTEXTUALISATION_MODEL"

# merge contextualised_chunks.jsonl into merged/contextualised_chunks.jsonl
MERGED_DIR="$PROCESSED_BASE/merged"
bash scripts/merge.sh "$PROCESSED_BASE" "$MERGED_DIR" contextualised_chunks.jsonl