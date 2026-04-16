#!/bin/bash

python lcr/generate_queries.py assure-queries \
    --queries-base-dir ./data/processed/polish_poc_more \
    --datasets queries \
    --save-path ./data/processed/polish_poc_more/checked_queries \
    --no-start-from-checkpoint \
    --llm "OpenAI/gpt-oss-120B" \
    --provider "together"

python lcr/visualisation/visualise_jsonl.py data/processed/polish_poc_more/assurance_results.jsonl --preset assurance_results \