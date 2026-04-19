import asyncio
from pathlib import Path

import typer

from lcr.anthropic_preprocessor import AnthropicContextualPreprocessor
from lcr.formatter import DataFormatter

app = typer.Typer()



async def contextualise_dataset_async(
    contextualisation_model: str = "qwen/qwen-2.5-7b-instruct",
    chunks_path: str = ".",
    save_dir: str = "./contextualised_datasets",
    provider: str = "openrouter",
    start_from_checkpoint: bool = False
):
    ds_formatter = DataFormatter()
    ds_formatter.load_from_jsonl(chunks_path, query_or_dataset="documents")
    preprocessor = AnthropicContextualPreprocessor(
        data_formatter=ds_formatter,
        contextualisation_model=contextualisation_model,
        provider=provider,
        save_dir=save_dir,
        start_from_checkpoint=start_from_checkpoint
    )
    await preprocessor.augment_documents()


@app.command()
def contextualise_dataset(
    contextualisation_model: str = "",
    chunks_path: str = ".",
    save_dir: str = "./contextualised_datasets",
    provider: str = "openrouter",
    start_from_checkpoint: bool = False
):
    """
    Contextualise a dataset using the specified model and provider.
    provider: 'openrouter' or 'together'
    """
    asyncio.run(contextualise_dataset_async(
        contextualisation_model, chunks_path, save_dir, provider, start_from_checkpoint
    ))


if __name__ == "__main__":
    app()
