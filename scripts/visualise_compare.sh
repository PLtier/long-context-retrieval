#!/bin/bash
QUERIES="${1:?Usage: $0 <queries.jsonl> <results_a.jsonl> <results_b.jsonl> [output.html] [--label-a LABEL] [--label-b LABEL] [--ignore-mismatch]}"
RESULTS_A="${2:?}"
RESULTS_B="${3:?}"
OUTPUT="${4:-}"

shift 4 2>/dev/null || shift $#

EXTRA_ARGS=("$@")

if [ -n "$OUTPUT" ]; then
    python3 lcr/visualisation/visualise_compare.py "$QUERIES" "$RESULTS_A" "$RESULTS_B" -o "$OUTPUT" "${EXTRA_ARGS[@]}"
else
    python3 lcr/visualisation/visualise_compare.py "$QUERIES" "$RESULTS_A" "$RESULTS_B" "${EXTRA_ARGS[@]}"
fi
