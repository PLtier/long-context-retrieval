#!/bin/bash


# Usage: ./update_queries.sh /path/to/dataset_dir /path/to/output_dir [prompt_template]
# Example: ./update_queries.sh ./data/processed/polish_dataset_7_no_chain/cluster_A/dataset_1 ./data/processed/polish_dataset_7_no_chain_is_questions/cluster_A/dataset_1 query_prompt_r8.j2


DATASET_DIR="$1"
OUTPUT_DIR="$2"
PROMPT_TEMPLATE="$3"


if [ -z "$DATASET_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 /path/to/dataset_dir /path/to/output_dir [prompt_template]"
    exit 1
fi


if [ -d "$DATASET_DIR" ] && [ -f "$DATASET_DIR/queries.jsonl" ]; then
    echo "Processing $DATASET_DIR -> $OUTPUT_DIR"
    PYTHON_CMD=(python lcr/cli/generate_queries.py generate-queries \
        --documents-base-dir "$DATASET_DIR" \
        --datasets chunks \
        --save-path "$OUTPUT_DIR" \
        --start-from-checkpoint \
        --llm "openai/gpt-oss-120b" \
        --provider "openrouter" \
        --impl-context-col "implicit_context_chunks" \
        --context-col "explicit_context_chunks" \
        --no-chain-context \
        --update-queries)
    if [ -n "$PROMPT_TEMPLATE" ]; then
        PYTHON_CMD+=(--prompt-template "$PROMPT_TEMPLATE")
    fi
    "${PYTHON_CMD[@]}"
    python lcr/visualisation/visualise_jsonl.py "$OUTPUT_DIR/queries.jsonl" --preset queries
else
    echo "Dataset directory or queries.jsonl not found: $DATASET_DIR"
    exit 2
fi
