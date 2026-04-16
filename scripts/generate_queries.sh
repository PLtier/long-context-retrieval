#!/bin/bash

python lcr/generate_queries.py generate-queries \
    --documents-base-dir ./data/interim/polish_poc_more \
    --datasets chunks \
    --save-path ./data/processed/polish_poc_more/generated_queries \
    --no-start-from-checkpoint \
    --llm "OpenAI/gpt-oss-120B" \
    --provider "together" \
    --impl-context-col "implicit_context_chunks_ids" \
    --context-col "explicit_context_chunks_ids" \
    --chain-context

python lcr/visualisation/visualise_jsonl.py data/processed/polish_poc_more/queries.jsonl --preset queries
