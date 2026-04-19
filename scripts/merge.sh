#!/bin/bash

# Merge all queries.jsonl and chunks.jsonl from all clusters into merged/queries.jsonl and merged/chunks.jsonl

QUERIES_BASE=./data/processed/danish_dataset_7_no_chain
MERGED_DIR="$QUERIES_BASE/merged"
CHUNKS_BASE=./data/raw/danish_dataset_7
mkdir -p "$MERGED_DIR"

# Merge queries.jsonl
echo "Merging queries.jsonl files..."
> "$MERGED_DIR/queries.jsonl"
for cluster in "$QUERIES_BASE"/cluster_*; do
	for dataset in "$cluster"/*; do
		if [ -d "$dataset" ] && [ -f "$dataset/queries.jsonl" ]; then
			echo "Adding $dataset/queries.jsonl"
			cat "$dataset/queries.jsonl" >> "$MERGED_DIR/queries.jsonl"
		fi
	done
done

# Merge chunks.jsonl
echo "Merging chunks.jsonl files..."
> "$MERGED_DIR/chunks.jsonl"
for cluster in "$CHUNKS_BASE"/cluster_*; do
	for dataset in "$cluster"/*; do
		if [ -d "$dataset" ] && [ -f "$dataset/chunks.jsonl" ]; then
			echo "Adding $dataset/chunks.jsonl"
			cat "$dataset/chunks.jsonl" >> "$MERGED_DIR/chunks.jsonl"
		fi
	done
done

echo "Merge complete. Output in $MERGED_DIR."
