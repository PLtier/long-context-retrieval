#!/bin/bash

QUERIES_PATH="$1"
DENSE_RESULTS_PATH="$2"
SPARSE_RESULTS_PATH="$3"
OUTPUT_PATH="$4"

python3 lcr/cli/filter_queries.py rank-filter \
    --queries-path "$QUERIES_PATH" \
    --dense-results-path "$DENSE_RESULTS_PATH" \
    --sparse-results-path "$SPARSE_RESULTS_PATH" \
    --output-path "$OUTPUT_PATH"

python lcr/visualisation/visualise_jsonl.py "$OUTPUT_PATH" --preset queries
