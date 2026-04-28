#!/bin/bash
# Usage: ./clean_contextualisation_failures.sh input.jsonl output.jsonl
# Removes all lines containing <CONTEXTUALISATION_FAILURE> from the input file and writes to output file.

set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input.jsonl output.jsonl"
  exit 1
fi

input_file="$1"
output_file="$2"

grep -v '<CONTEXTUALISATION_FAILURE>' "$input_file" > "$output_file"

echo "Cleaned file written to $output_file"
