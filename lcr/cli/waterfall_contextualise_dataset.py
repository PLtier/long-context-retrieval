import asyncio

import typer

from lcr.formatter import DataFormatter
from lcr.waterfall_preprocessor import WaterfallContextualPreprocessor

app = typer.Typer()


async def waterfall_contextualise_dataset_async(
    contextualisation_model: str,
    chunks_path: str,
    save_dir: str,
    provider: str,
    start_from_checkpoint: bool,
    max_concurrent: int,
    window_tokens: int,
    overlap_tokens: int,
    aggregation: str,
    tokenizer_name: str,
):
    ds_formatter = DataFormatter()
    ds_formatter.load_from_jsonl(chunks_path, query_or_dataset="documents")
    preprocessor = WaterfallContextualPreprocessor(
        data_formatter=ds_formatter,
        contextualisation_model=contextualisation_model,
        provider=provider,
        save_dir=save_dir,
        start_from_checkpoint=start_from_checkpoint,
        max_concurrent=max_concurrent,
        window_tokens=window_tokens,
        overlap_tokens=overlap_tokens,
        aggregation=aggregation,
        tokenizer_name=tokenizer_name,
    )
    await preprocessor.augment_documents()


@app.command()
def waterfall_contextualise_dataset(
    contextualisation_model: str = "",
    chunks_path: str = ".",
    save_dir: str = "./contextualised_datasets",
    provider: str = "vllm",
    start_from_checkpoint: bool = False,
    max_concurrent: int = 50,
    window_tokens: int = 32_000,
    overlap_tokens: int = 8_000,
    aggregation: str = "append",
    tokenizer_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
):
    """
    Waterfall contextualise: split each document into overlapping token windows
    and contextualise every chunk against every window. Aggregates per-chunk
    contexts via 'append' (concatenate) or 'consolidate' (extra LLM merge pass).

    provider: 'openrouter', 'together', or 'vllm' (default: 'vllm' for local Qwen3)
    aggregation: 'append' or 'consolidate'
    """
    if aggregation not in ("append", "consolidate"):
        raise typer.BadParameter("aggregation must be 'append' or 'consolidate'")
    asyncio.run(
        waterfall_contextualise_dataset_async(
            contextualisation_model,
            chunks_path,
            save_dir,
            provider,
            start_from_checkpoint,
            max_concurrent,
            window_tokens,
            overlap_tokens,
            aggregation,
            tokenizer_name,
        )
    )


if __name__ == "__main__":
    app()
