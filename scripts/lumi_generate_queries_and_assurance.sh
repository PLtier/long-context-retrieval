#!/bin/bash
cd ~/repositories/long-context-retrieval
source .venv/bin/activate

RAW_BASE="./data/raw/polish_final_cluster_reparsed"
PROCESSED_BASE="./data/processed/polish_final_cluster_reparsed"
QUERY_PROMPT_TEMPLATE="query_generation_prompt_r9.j2"
ASSURANCE_PROMPT_TEMPLATE="assurance_prompt_r10.j2"


DK_SPACY_MODEL="da_core_news_sm"
PL_SPACY_MODEL="pl_core_news_sm"
if [[ "$RAW_BASE" == *"danish"* ]]; then
    SPACY_MODEL="$DK_SPACY_MODEL"
elif [[ "$RAW_BASE" == *"polish"* ]]; then
    SPACY_MODEL="$PL_SPACY_MODEL"
else
    echo "Error: unrecognized dataset in RAW_BASE path. Please set SPACY_MODEL manually."
    exit 1
fi

# ssh -N -L 8000:nid005160:8000 jalocham@lumi.csc.fi &

# echo "Generating queries with VLLM LLM..."
# VLLM_BASE_URL=http://localhost:8000/v1 bash ./scripts/generate_queries.sh "$RAW_BASE" "$PROCESSED_BASE" "$QUERY_PROMPT_TEMPLATE"

MERGED_DIR="$PROCESSED_BASE/merged"

# # Merge per-cluster outputs into a single file each.
# # Queries and augmented_chunks live under processed; raw chunks live under raw.
# # Assurance is intentionally not merged here — it runs later on the hard subset only.
# echo "Merging per-cluster outputs into $MERGED_DIR..."
# bash scripts/merge.sh "$PROCESSED_BASE" "$MERGED_DIR" queries.jsonl
# bash scripts/merge.sh "$PROCESSED_BASE" "$MERGED_DIR" augmented_chunks.jsonl
# bash scripts/merge.sh "$RAW_BASE"       "$MERGED_DIR" chunks.jsonl

# # Drop query-generation failures (rows with `query=="<QUERY_GENERATION_FAILURE>"`).
# echo "Filtering out query generation failures..."
# python3 lcr/cli/filter_queries.py drop-failures \
#     --queries-path "$MERGED_DIR/queries.jsonl" \
#     --output-path "$MERGED_DIR/successful_queries.jsonl"

# # Stage 1: dense (bge-m3) + sparse (BM25) retrieval over plain chunks, with all queries
# echo "Running dense and sparse retrieval evaluation on all queries..."
# bash scripts/eval_gpu.sh \
#     "$MERGED_DIR/chunks_bge-m3_dense_results.jsonl" \
#     "$MERGED_DIR/chunks.jsonl" \
#     "$MERGED_DIR/successful_queries.jsonl" \
#     dense

# bash scripts/eval_bm25.sh \
#     "$MERGED_DIR/chunks_bm25_results.jsonl" \
#     "$MERGED_DIR/chunks.jsonl" \
#     "$MERGED_DIR/successful_queries.jsonl" \
#     "$SPACY_MODEL"

# Stage 2a:rank-filter — keep queries hard for both retrievers (saves LLM cost on Stage 2b)
# echo "Filtering queries to find hard subset for assurance..."
# bash scripts/rank_filter_queries.sh \
#     "$MERGED_DIR/successful_queries.jsonl" \
#     "$MERGED_DIR/chunks_bge-m3_dense_results.jsonl" \
#     "$MERGED_DIR/chunks_bm25_results.jsonl" \
#     "$MERGED_DIR/hard_queries.jsonl"

# # Stage 2b: run assurance only on the hard queries
# echo "Running assurance on hard queries..."
# VLLM_BASE_URL=http://localhost:8000/v1 python lcr/cli/generate_queries.py assure-queries \
#     --queries-base-dir "$MERGED_DIR" \
#     --datasets hard_queries \
#     --save-path "$MERGED_DIR" \
#     --llm "openai/gpt-oss-120b" \
#     --provider "vllm" \
#     --prompt-template "$ASSURANCE_PROMPT_TEMPLATE" \
#     --start-from-checkpoint
# python lcr/visualisation/visualise_jsonl.py "$MERGED_DIR/assurance_results.jsonl" --preset assurance_results

# # Stage 2c: assurance-filter — drop hard queries that didn't pass assurance
# echo "Filtering hard queries to find contextual subset for final evaluation..."
# bash scripts/assurance_filter_queries.sh \
#     "$MERGED_DIR/hard_queries.jsonl" \
#     "$MERGED_DIR/assurance_results.jsonl" \
#     "$MERGED_DIR/contextual_queries.jsonl"

# # Stage 3: dense + sparse retrieval over augmented chunks, contextual queries only
# echo "Running dense and sparse retrieval evaluation on contextual queries and augmented chunks..."
# bash scripts/eval_gpu.sh \
#     "$MERGED_DIR/augmented_chunks_bge-m3_dense_results.jsonl" \
#     "$MERGED_DIR/augmented_chunks.jsonl" \
#     "$MERGED_DIR/contextual_queries.jsonl" \
#     dense

# bash scripts/eval_bm25.sh \
#     "$MERGED_DIR/augmented_chunks_bm25_results.jsonl" \
#     "$MERGED_DIR/augmented_chunks.jsonl" \
#     "$MERGED_DIR/contextual_queries.jsonl" \
#     "$SPACY_MODEL"
