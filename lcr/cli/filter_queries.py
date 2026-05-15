import json
import os
import re

from loguru import logger
import typer

app = typer.Typer()


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_rank_lookup(results: list[dict]) -> dict[tuple[str, str], int | None]:
    return {
        (r["query"], r["gt_chunk_id"]): r.get("gt_chunk_rank")
        for r in results
    }


def _in_top_k(rank: int | None, top_k: int) -> bool:
    return rank is not None and rank <= top_k


@app.command("drop-failures")
def drop_failures(
    queries_path: str = typer.Option(..., "--queries-path"),
    output_path: str = typer.Option(..., "--output-path"),
):
    """Drop rows where query generation failed (query == '<QUERY_GENERATION_FAILURE>').

    These rows have heterogeneous schemas (e.g. utilized_context_chunk_ids becomes a
    string instead of a list) which breaks HuggingFace `datasets` loading downstream.
    """
    queries = _load_jsonl(queries_path)
    kept = [r for r in queries if r.get("query") != "<QUERY_GENERATION_FAILURE>"]

    dropped = len(queries) - len(kept)
    logger.info(f"input queries:   {len(queries)}")
    logger.info(f"dropped (failed):{dropped}")

    # also, an addon for polish dataset: drop queries for target chunks with end with ":" or start with "-"
    def startswith_hyphen(chunk: str) -> bool:
        """checks if the chunk starts with: ^(Art\.)? ?([\d\w]+)?\.? ?([\d\w]+[\.\)])? ?-"""
        pattern = r"^(Art\.)? ?([\d\w]+)?\.? ?([\d\w]+[\.\)])? ?-"
        return re.match(pattern, chunk) is not None

    if "polish" in queries_path.lower():
        before = len(kept)
        kept = [r for r in kept if not (r["chunk"].endswith(":") or startswith_hyphen(r["chunk"]))]
        logger.info(f"additionally dropped {before - len(kept)} queries with invalid chunk_ids")

    logger.info(f"kept:            {len(kept)}")
    _write_jsonl(output_path, kept)
    logger.info(f"wrote {output_path}")


@app.command("rank-filter")
def rank_filter(
    queries_path: str = typer.Option(..., "--queries-path"),
    dense_results_path: str = typer.Option(..., "--dense-results-path"),
    sparse_results_path: str = typer.Option(..., "--sparse-results-path"),
    output_path: str = typer.Option(..., "--output-path"),
    top_k: int = typer.Option(10, "--top-k"),
):
    """Keep queries whose GT chunk is NOT in top-k for either dense or sparse."""
    queries = _load_jsonl(queries_path)
    dense_rank = _build_rank_lookup(_load_jsonl(dense_results_path))
    sparse_rank = _build_rank_lookup(_load_jsonl(sparse_results_path))

    dropped_dense = 0
    dropped_sparse = 0
    kept: list[dict] = []
    for row in queries:
        key = (row["query"], row["chunk_id"])
        if _in_top_k(dense_rank.get(key), top_k):
            dropped_dense += 1
            continue
        if _in_top_k(sparse_rank.get(key), top_k):
            dropped_sparse += 1
            continue
        kept.append(row)

    logger.info(f"input queries:        {len(queries)}")
    logger.info(f"dropped (dense top{top_k}): {dropped_dense}")
    logger.info(f"dropped (sparse top{top_k}):{dropped_sparse}")
    logger.info(f"kept (hard):          {len(kept)}")
    _write_jsonl(output_path, kept)
    logger.info(f"wrote {output_path}")


@app.command("assurance-filter")
def assurance_filter(
    queries_path: str = typer.Option(..., "--queries-path"),
    assurance_path: str = typer.Option(..., "--assurance-path"),
    output_path: str = typer.Option(..., "--output-path"),
):
    """Keep queries whose GT chunk_id has passes_assurance == 'Yes'."""
    queries = _load_jsonl(queries_path)
    assurance = {r["chunk_id"]: r.get("passes_assurance") for r in _load_jsonl(assurance_path)}

    dropped = 0
    kept: list[dict] = []
    for row in queries:
        if assurance.get(row["chunk_id"]) != "Yes":
            dropped += 1
            continue
        kept.append(row)

    logger.info(f"input queries:        {len(queries)}")
    logger.info(f"dropped (assurance):  {dropped}")
    logger.info(f"kept (contextual):    {len(kept)}")
    _write_jsonl(output_path, kept)
    logger.info(f"wrote {output_path}")


if __name__ == "__main__":
    app()
