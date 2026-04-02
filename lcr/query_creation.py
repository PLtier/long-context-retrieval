from lcr.query_generator import QueryGenerator
from aiohttp.typedefs import Query
import asyncio
from sympy.assumptions.handlers.calculus import _
from pathlib import Path

from loguru import logger
import typer

from lcr.config import PROCESSED_DATA_DIR
from lcr.formatter import DataFormatter
from lcr.utils import DATASETS

app = typer.Typer()

async def _generate_queries(
    documents_base_dir: Path,
    datasets: list[str],
    save_path: str = "./generated_queries",
    llm: str = "qwen/qwen-2.5-7b-instruct",
    start_from_checkpoint: bool = True,
    save_jsonl: bool = True,
):
    for dataset in datasets:
        path = DATASETS[dataset].get("path", dataset)

        ds_formatter = DataFormatter()
        ds_formatter.load_from_jsonl(f"{documents_base_dir}/{path}.jsonl", query_or_dataset="documents")

        queries_generator = QueryGenerator(
            ds_formatter,
            llm,
            save_path,
            start_from_checkpoint=start_from_checkpoint,
            save_jsonl=save_jsonl,
        )
        await queries_generator.generate_queries()
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
    
    # -----------------------------------------
):
    asyncio.run(_generate_queries(documents_base_dir, datasets, save_path, llm, start_from_checkpoint, save_jsonl))
           



if __name__ == "__main__":
    app()