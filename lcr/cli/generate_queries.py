import asyncio
from pathlib import Path

from loguru import logger
import typer

from lcr.config import PROCESSED_DATA_DIR
from lcr.formatter import DataFormatter
from lcr.query_generator import QueryAssurance, QueryGenerator
from lcr.utils import DATASETS

app = typer.Typer()

async def _generate_queries(
    documents_base_dir: Path,
    datasets: list[str],
    save_path: str = "",
    llm: str = "",
    provider: str = "",
    start_from_checkpoint: bool = True,
    save_jsonl: bool = True,
    chain_context: bool = False,
    impl_context_col: str = "implicit_context_chunks_ids",
    context_col: str = "context_chunks_ids",
    update_queries: bool = False,
    prompt_template: str = "",
):
    if (not (documents_base_dir and save_path and llm and provider and prompt_template)):
        logger.info("Missing crucial argument for query generation. Please provide all required arguments.")
    for dataset in datasets:
        path = DATASETS[dataset].get("path", dataset)
        print(path)

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
            update_queries=update_queries,
            prompt_template=prompt_template,
            input_queries_dir=str(documents_base_dir),
        )
        await queries_generator.generate(chain_context=chain_context)
        print(f"Queries for dataset {dataset} generated and saved to {save_path}")

@app.command()
def generate_queries(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    documents_base_dir: Path = PROCESSED_DATA_DIR,
    datasets: list[str] = [],
    save_path: str = "",
    llm: str = "",
    start_from_checkpoint: bool = True,
    save_jsonl: bool = True,
    provider: str = "",
    chain_context: bool = False,
    impl_context_col: str = "implicit_context_chunks_ids",
    context_col: str = "context_chunks_ids",
    update_queries: bool = False,
    
    prompt_template: str = "",
    # -----------------------------------------
):
    asyncio.run(_generate_queries(documents_base_dir, datasets, save_path, llm, provider, start_from_checkpoint, save_jsonl, chain_context, impl_context_col, context_col, update_queries, prompt_template))

@app.command()
def assure_queries(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    queries_base_dir: Path = PROCESSED_DATA_DIR,
    datasets: list[str] = [],
    save_path: str = "",
    llm: str = "",
    start_from_checkpoint: bool = True,
    save_jsonl: bool = True,
    provider: str = "",
    
    prompt_template: str = "",
    # -----------------------------------------
):
    if (not (queries_base_dir and save_path and llm and provider and prompt_template)):
        logger.info("Missing crucial argument for query assurance. Please provide all required arguments.")
    for dataset in datasets:
        ds_formatter = DataFormatter()
        ds_formatter.load_from_jsonl(f"{queries_base_dir}/{dataset}.jsonl", query_or_dataset="queries")

        assurance = QueryAssurance(
            ds_formatter,
            llm,
            provider,
            save_path,
            start_from_checkpoint=start_from_checkpoint,
            save_jsonl=save_jsonl,
            prompt_template=prompt_template
        )
        asyncio.run(assurance.generate())
        print(f"Assurance for dataset {dataset} generated and saved to {save_path}")     



if __name__ == "__main__":
    app()