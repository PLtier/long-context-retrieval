#!/bin/bash

RESULTS_JSONL_PATH="$1"
CHUNKS_PATH="$2"
QUERIES_PATH="$3"
SPACY_MODEL="$4"

echo 
python3 lcr/cli/eval.py \
    --chunks-path "$CHUNKS_PATH" \
    --results-jsonl-path "$RESULTS_JSONL_PATH" \
    --queries-path "$QUERIES_PATH" \
    --no-use-prefix \
    --encoder "bm25" \
    --spacy-model "$SPACY_MODEL"