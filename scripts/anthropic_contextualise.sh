
#!/bin/bash

# Contextualise merged/chunks.jsonl into merged/contextualised_chunks.jsonl using anthropic_contextualise_dataset.py

CHUNKS_PATH="./data/processed/danish_dataset_7_no_chain/merged/chunks.jsonl"
SAVE_DIR="./data/processed/danish_dataset_7_no_chain/merged"
CONTEXTUALISATION_MODEL="OpenAI/gpt-oss-120B"  # or leave blank for default

python3 lcr/cli/anthropic_contextualise_dataset.py \
    --contextualisation-model "$CONTEXTUALISATION_MODEL" \
    --chunks-path "$CHUNKS_PATH" \
    --save-dir "$SAVE_DIR" \
    --provider "together" \
    --no-start-from-checkpoint
