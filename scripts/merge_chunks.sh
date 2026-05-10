#!/bin/bash

# Usage: ./merge_chunks.sh <input_dir> <output_dir>
# Merges all chunks.jsonl files from <input_dir>/cluster_*/<dataset>/chunks.jsonl into <output_dir>/chunks.jsonl

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_dir> <output_dir> <chunks_filename>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CHUNKS_FILENAME="$3"
MERGED_CHUNKS="$OUTPUT_DIR/$CHUNKS_FILENAME"

mkdir -p "$OUTPUT_DIR"

# Merge chunks.jsonl
> "$MERGED_CHUNKS"
echo "Merging chunks.jsonl files from $INPUT_DIR to $MERGED_CHUNKS..."
for cluster in "$INPUT_DIR"/cluster_*; do
    for dataset in "$cluster"/*; do
        if [ -d "$dataset" ] && [ -f "$dataset/$CHUNKS_FILENAME" ]; then
            echo "Adding $dataset/$CHUNKS_FILENAME"
            cat "$dataset/$CHUNKS_FILENAME" >> "$MERGED_CHUNKS"
            echo >> "$MERGED_CHUNKS" # Add a newline between files
        fi
    done
done
echo "Merge complete. Output in $MERGED_CHUNKS."
