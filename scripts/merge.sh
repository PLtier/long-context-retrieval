#!/bin/bash

# Usage: ./merge.sh <input_dir> <output_dir> <filename>
# Concatenates every <input_dir>/cluster_*/<dataset>/<filename> into <output_dir>/<filename>.

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_dir> <output_dir> <filename>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
FILENAME="$3"
MERGED="$OUTPUT_DIR/$FILENAME"

mkdir -p "$OUTPUT_DIR"
> "$MERGED"

echo "Merging $FILENAME from $INPUT_DIR to $MERGED..."
for cluster in "$INPUT_DIR"/cluster_*; do
    for dataset in "$cluster"/*; do
        if [ -d "$dataset" ] && [ -f "$dataset/$FILENAME" ]; then
            echo "Adding $dataset/$FILENAME"
            cat "$dataset/$FILENAME" >> "$MERGED"
            echo >> "$MERGED"
        fi
    done
done
echo "Merge complete. Output in $MERGED."
