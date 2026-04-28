#!/bin/bash

# Usage: ./merge_chunks.sh <input_dir> <output_dir>
# Merges all chunks.jsonl files from <input_dir>/cluster_*/<dataset>/chunks.jsonl into <output_dir>/chunks.jsonl

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
MERGED_CHUNKS="$OUTPUT_DIR/chunks.jsonl"

mkdir -p "$OUTPUT_DIR"

# Merge chunks.jsonl
> "$MERGED_CHUNKS"
echo "Merging chunks.jsonl files from $INPUT_DIR to $MERGED_CHUNKS..."
for cluster in "$INPUT_DIR"/cluster_*; do
    for dataset in "$cluster"/*; do
        if [ -d "$dataset" ] && [ -f "$dataset/chunks.jsonl" ]; then
            echo "Adding $dataset/chunks.jsonl"
            cat "$dataset/chunks.jsonl" >> "$MERGED_CHUNKS"
            echo >> "$MERGED_CHUNKS" # Add a newline between files
        fi
    done
done
echo "Merge complete. Output in $MERGED_CHUNKS."
