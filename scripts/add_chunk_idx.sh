#!/bin/bash
# Usage: bash scripts/add_chunk_idx.sh input.jsonl [output.jsonl]
# Adds chunk_idx field = last segment of chunk_id after splitting on '_'.
# Writes to output.jsonl if given, otherwise overwrites input in-place.

INPUT="$1"
OUTPUT="${2:-}"

if [[ -z "$INPUT" ]]; then
    echo "Usage: $0 input.jsonl [output.jsonl]" >&2
    exit 1
fi

DEST="${OUTPUT:-$INPUT}"
TMP=$(mktemp)
jq -sc '[.[] | . + {chunk_idx: (.chunk_id | split("_") | last)}] | sort_by(.chunk_idx | tonumber) | .[]' "$INPUT" > "$TMP" && mv "$TMP" "$DEST"
