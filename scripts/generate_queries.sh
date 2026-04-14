#!/bin/bash

python lcr/generate_queries.py generate-queries \
    --documents-base-dir ./data/interim/danish_dataset_short_poc \
    --datasets chunks \
    --save-path ./data/processed/danish_dataset_short_poc/generated_queries \
    --no-start-from-checkpoint \
    --llm "OpenAI/gpt-oss-120B" \
    --provider "together" \
    --impl-context-col "implicit_context_chunks_ids" \
    --context-col "explicit_context_chunks_ids" \