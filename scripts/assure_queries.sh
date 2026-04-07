#!/bin/bash

python lcr/query_creation.py assure-queries \
    --queries-base-dir ./data/processed/poc_link_queries_1_context_gpt_oss/ \
    --datasets queries \
    --save-path ./data/processed/poc_link_queries_1_context_gpt_oss/checked_queries \
    --no-start-from-checkpoint \
    --llm "OpenAI/gpt-oss-20B" \
    --provider "together" \