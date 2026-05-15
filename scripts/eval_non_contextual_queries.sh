#!/bin/bash
set -euo pipefail
cd ~/repositories/long-context-retrieval
source .venv/bin/activate

EMBEDDER_MODEL="BAAI/bge-m3"
POLISH_DIR="./data/processed/polish_final_cluster_reparsed/merged"
DANISH_DIR="./data/processed/danish_final_cluster_r10/merged"

# Step 1 — build non_contextual_queries.jsonl per language as
#          successful_queries minus contextual_queries (keyed on chunk_id+query)
build_non_contextual() {
  local DIR="$1"
  python3 - "$DIR" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
ctx = {(r["chunk_id"], r["query"])
       for r in map(json.loads, (d / "contextual_queries.jsonl").open())}
n_in = n_out = 0
with (d / "non_contextual_queries.jsonl").open("w") as out:
    for line in (d / "successful_queries.jsonl").open():
        n_in += 1
        r = json.loads(line)
        if (r["chunk_id"], r["query"]) in ctx:
            continue
        out.write(line); n_out += 1
print(f"  {d}: {n_in} successful - {len(ctx)} contextual -> {n_out} non-contextual")
PY
}
echo "Building non_contextual_queries.jsonl"
build_non_contextual "$POLISH_DIR"
build_non_contextual "$DANISH_DIR"

# Step 2 — 2x2 eval sweep
run_eval() {
  local CHUNKS="$1"; local QUERIES="$2"; local OUT="$3"
  mkdir -p "$(dirname "$OUT")"
  python3 lcr/cli/eval.py \
    --chunks-path "$CHUNKS" \
    --model-path "$EMBEDDER_MODEL" \
    --results-jsonl-path "$OUT" \
    --queries-path "$QUERIES" \
    --no-use-prefix \
    --encoder "flag" \
    --encoding-type "dense" \
    --batch-size 32
}

echo "Polish | normal chunks | non_contextual_queries"
run_eval "$POLISH_DIR/chunks.jsonl"           "$POLISH_DIR/non_contextual_queries.jsonl" "$POLISH_DIR/results/bge-m3_chunks_non_contextual.jsonl"

echo "Polish | augmented chunks | non_contextual_queries"
run_eval "$POLISH_DIR/augmented_chunks.jsonl" "$POLISH_DIR/non_contextual_queries.jsonl" "$POLISH_DIR/results/bge-m3_augmented_non_contextual.jsonl"

echo "Danish | normal chunks | non_contextual_queries"
run_eval "$DANISH_DIR/chunks.jsonl"           "$DANISH_DIR/non_contextual_queries.jsonl" "$DANISH_DIR/results/bge-m3_chunks_non_contextual.jsonl"

echo "Danish | augmented chunks | non_contextual_queries"
run_eval "$DANISH_DIR/augmented_chunks.jsonl" "$DANISH_DIR/non_contextual_queries.jsonl" "$DANISH_DIR/results/bge-m3_augmented_non_contextual.jsonl"
