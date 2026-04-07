#!/bin/bash

python lcr/query_creation.py generate-queries \
    --documents-base-dir ./data/processed/poc_link_queries_1_context_gpt_oss \
    --datasets chunks \
    --save-path ./data/processed/poc_link_queries_1_context_gpt_oss/generated_queries \
    --no-start-from-checkpoint \
    --llm "OpenAI/gpt-oss-20B" \
    --provider "together" \