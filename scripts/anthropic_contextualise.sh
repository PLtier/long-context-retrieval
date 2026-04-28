
#!/bin/bash

# Contextualise merged/chunks.jsonl into merged/contextualised_chunks.jsonl using anthropic_contextualise_dataset.py

CHUNKS_PATH="$1"
SAVE_DIR="$2"
CONTEXTUALISATION_MODEL=# or leave blank for default

python3 lcr/cli/anthropic_contextualise_dataset.py \
    --contextualisation-model "$CONTEXTUALISATION_MODEL" \
    --chunks-path "$CHUNKS_PATH" \
    --save-dir "$SAVE_DIR" \
    --provider "together" \
    --no-start-from-checkpoint
