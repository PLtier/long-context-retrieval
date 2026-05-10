#!/bin/bash

QUERIES_PATH="$1"
ASSURANCE_PATH="$2"
OUTPUT_PATH="$3"

python3 lcr/cli/filter_queries.py assurance-filter \
    --queries-path "$QUERIES_PATH" \
    --assurance-path "$ASSURANCE_PATH" \
    --output-path "$OUTPUT_PATH"

python lcr/visualisation/visualise_jsonl.py "$OUTPUT_PATH" --preset queries
