#!/bin/bash

python lcr/generate_queries.py assure-queries \
    --queries-base-dir ./data/processed/danish_dataset_short_poc/ \
    --datasets queries \
    --save-path ./data/processed/danish_dataset_short_poc/checked_queries \
    --no-start-from-checkpoint \
    --llm "OpenAI/gpt-oss-20B" \
    --provider "together" \