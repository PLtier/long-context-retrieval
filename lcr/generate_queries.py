import asyncio
from pathlib import Path

import typer

from lcr.config import PROCESSED_DATA_DIR
from lcr.formatter import DataFormatter
from lcr.query_generator import QueryAssurance, QueryGenerator
from lcr.utils import DATASETS

app = typer.Typer()

async def _generate_queries(
    documents_base_dir: Path,
    datasets: list[str],
    save_path: str = "./generated_queries",
    llm: str = "qwen/qwen-2.5-7b-instruct",
    provider: str = "openrouter",
    start_from_checkpoint: bool = True,
    save_jsonl: bool = True,
    chain_context: bool = False,
    impl_context_col: str = "implicit_context_chunks_ids",
    context_col: str = "context_chunks_ids",
):
    for dataset in datasets:
        path = DATASETS[dataset].get("path", dataset)

        ds_formatter = DataFormatter()
        ds_formatter.load_from_jsonl(f"{documents_base_dir}/{path}.jsonl", query_or_dataset="documents")

        queries_generator = QueryGenerator(
            ds_formatter,
            llm,
            provider,
            save_path,
            start_from_checkpoint=start_from_checkpoint,
            save_jsonl=save_jsonl,
            impl_context_col=impl_context_col,
            context_col=context_col,
        )
        await queries_generator.generate(chain_context=chain_context)
        print(f"Queries for dataset {dataset} generated and saved to {save_path}")

@app.command()
def generate_queries(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    documents_base_dir: Path = PROCESSED_DATA_DIR,
    datasets: list[str] = [],
    save_path: str = "./generated_queries",
    llm: str = "qwen/qwen-2.5-7b-instruct",
    start_from_checkpoint: bool = True,
    save_jsonl: bool = True,
    provider: str = "openrouter",
    chain_context: bool = False,
    impl_context_col: str = "implicit_context_chunks_ids",
    context_col: str = "context_chunks_ids",
    
    # -----------------------------------------
):
    asyncio.run(_generate_queries(documents_base_dir, datasets, save_path, llm, provider, start_from_checkpoint, save_jsonl, chain_context, impl_context_col, context_col))

@app.command()
def assure_queries(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    queries_base_dir: Path = PROCESSED_DATA_DIR,
    datasets: list[str] = [],
    save_path: str = "./assurance_results",
    llm: str = "qwen/qwen-2.5-7b-instruct",
    start_from_checkpoint: bool = True,
    save_jsonl: bool = True,
    provider: str = "openrouter",
    
    # -----------------------------------------
):
    for dataset in datasets:
        # path = DATASETS[dataset].get("path", dataset)
        ds_formatter = DataFormatter()
        ds_formatter.load_from_jsonl(f"{queries_base_dir}/{dataset}.jsonl", query_or_dataset="queries")

        assurance = QueryAssurance(
            ds_formatter,
            llm,
            provider,
            save_path,
            start_from_checkpoint=start_from_checkpoint,
            save_jsonl=save_jsonl,
        )
        asyncio.run(assurance.generate())
        print(f"Assurance for dataset {dataset} generated and saved to {save_path}")     



if __name__ == "__main__":
    app()