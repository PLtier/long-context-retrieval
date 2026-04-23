from operator import is_
from pathlib import Path

from lcr.anthropic_preprocessor import AnthropicContextualPreprocessor
from lcr.formatter import DataFormatter

DATASETS: dict[str, dict] = {
    # "football": {},
    # "geography": {},
    # "mldr-conteb-eval": {"split": "test"},
    # "covid-qa": {},
    # "esg-reports": {"split": "test"},
    # "narrative-qa": {"split": "train"},
    # "squad-conteb-eval": {"split": "validation"},
    # "squad-conteb-train": {"split": "train"},
    "insurance": {},
    "example": {"is_query_local": True, "is_docs_local": True, "split": None},
    "chunks": {"is_query_local": True, "is_docs_local": True, "split": None},
    "queries": {"is_query_local": True, "is_docs_local": True, "split": None},
}

async def contextualise_datasets_async(
    contextualisation_model: str = "qwen/qwen-2.5-7b-instruct",
    datasets: list[str] = list(DATASETS.keys()),
    data_base_path: str = ".",
    save_dir: str = "./contextualised_datasets",
):
    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"Dataset {dataset} not found in DATASETS. Skipping.")
            continue
        path = DATASETS[dataset].get("path", dataset)
        ds_path = f"{data_base_path}/{path}"
        split = DATASETS[dataset].get("split", "train")
        is_query_local = DATASETS[dataset].get("is_query_local", False)
        is_docs_local = DATASETS[dataset].get("is_docs_local", False)
        ds_formatter = DataFormatter()
        ds_formatter.load_queries(ds_path, is_query_local, split)
        ds_formatter.load_documents(ds_path, is_docs_local, split)
        preprocessor = AnthropicContextualPreprocessor(
            data_formatter=ds_formatter,
            contextualisation_model=contextualisation_model,
        )
        # ensure that the path is writeable before we start the async processing by saving an empty file there (this will error if the path is not writeable, preventing us from doing a long async run only to find out at the end that we can't save the results)
        await preprocessor.augment_documents()
        save_path = f"{save_dir}/{dataset}"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        preprocessor.save_augmented_documents(save_path)
        print(f"Contextualised dataset saved to {save_path}")