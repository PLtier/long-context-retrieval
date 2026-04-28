#!/bin/bash

# Set the directory containing the .jsonl files
SOURCE_DIR="$1"

# Iterate through all .jsonl files in the directory
for file in "$SOURCE_DIR"/*.jsonl; do
    # Get the base filename without extension
    filename=$(basename "$file" .jsonl)
    # Create a new directory named after the file
    mkdir -p "$SOURCE_DIR/$filename"
    # Move and rename the file into the new directory
    mv "$file" "$SOURCE_DIR/$filename/chunks.jsonl"
done