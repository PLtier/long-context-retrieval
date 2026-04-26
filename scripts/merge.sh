#!/bin/bash

# Merge all queries.jsonl and chunks.jsonl from all clusters into merged/queries.jsonl and merged/chunks.jsonl

QUERIES_BASE=./data/processed/danish_dataset_7_no_chain
MERGED_DIR="$QUERIES_BASE/merged"
CHUNKS_BASE=./data/raw/danish_dataset_7
ASSURANCE_BASE=./data/processed/danish_dataset_7_no_chain
mkdir -p "$MERGED_DIR"

# # Merge queries.jsonl
# echo "Merging queries.jsonl files..."
# > "$MERGED_DIR/queries.jsonl"
# for cluster in "$QUERIES_BASE"/cluster_*; do
# 	for dataset in "$cluster"/*; do
# 		if [ -d "$dataset" ] && [ -f "$dataset/queries.jsonl" ]; then
# 			echo "Adding $dataset/queries.jsonl"
# 			cat "$dataset/queries.jsonl" >> "$MERGED_DIR/queries.jsonl"
# 			echo >> "$MERGED_DIR/queries.jsonl" # Add a newline between files
# 		fi
# 	done
# done

# run visualisation on merged queries
python lcr/visualisation/visualise_jsonl.py "$MERGED_DIR/queries.jsonl" --preset queries

# Merge chunks.jsonl
echo "Merging chunks.jsonl files..."
> "$MERGED_DIR/chunks.jsonl"
for cluster in "$CHUNKS_BASE"/cluster_*; do
	for dataset in "$cluster"/*; do
		if [ -d "$dataset" ] && [ -f "$dataset/chunks.jsonl" ]; then
			echo "Adding $dataset/chunks.jsonl"
			cat "$dataset/chunks.jsonl" >> "$MERGED_DIR/chunks.jsonl"
			echo >> "$MERGED_DIR/chunks.jsonl" # Add a newline between files
		fi
	done
done

# # Merge assurance_results.jsonl
# echo "Merging assurance_results.jsonl files..."
# > "$MERGED_DIR/assurance_results.jsonl"
# for cluster in "$ASSURANCE_BASE"/cluster_*; do
# 	for dataset in "$cluster"/*; do
# 		if [ -d "$dataset" ] && [ -f "$dataset/assurance_results.jsonl" ]; then
# 			echo "Adding $dataset/assurance_results.jsonl"
# 			cat "$dataset/assurance_results.jsonl" >> "$MERGED_DIR/assurance_results.jsonl"
# 			echo >> "$MERGED_DIR/assurance_results.jsonl" # Add a newline between files
# 		fi
# 	done
# done

# visualise
python lcr/visualisation/visualise_jsonl.py "$MERGED_DIR/assurance_results.jsonl" --preset assurance_results

echo "Merge complete. Output in $MERGED_DIR."
