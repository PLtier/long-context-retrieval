from typing import Generator

from datasets import load_dataset
from datasets.load import load_from_disk
import pandas as pd


class DataFormatter():
    def __init__(
        self,
    ):
        self.doc_dataset = None
        self.queries_dataset = None
        self.query_prompt = ""
        self.doc_prompt = ""

    def load_queries(self, path: str, is_local: bool, split: str = "", name="queries") -> None:
        if is_local:
            self.queries_dataset = load_from_disk(path)
            if split:
                self.queries_dataset = self.queries_dataset[split]
            print("Loaded queries dataset from disk", path, "split:", split, "name:", name)
        else:
            self.queries_dataset = load_dataset(path, split=split, name=name)
            print("Loaded queries dataset from hf", path, "split:", split, "name:", name)
        self.queries_dataset = self.queries_dataset.map(self.parse_id)

    def load_documents(self, path: str, is_local: bool, split: str = "", name: str = "documents") -> None:
        if is_local:
            self.doc_dataset = load_from_disk(path)
            if split:
                self.doc_dataset = self.doc_dataset[split]
            print("Loaded documents dataset from disk", path, "split:", split, "name:", name)
        else:
            self.doc_dataset = load_dataset(path, split=split, name=name)
            print("Loaded documents dataset from hf", path, "split:", split, "name:", name)
        self.doc_dataset = self.doc_dataset.map(self.parse_id)
    
    def load_from_jsonl(self, path: str, query_or_dataset: str) -> None:
        """Loads a query or document dataset from a jsonl file."""
        if query_or_dataset == "queries":
            self.queries_dataset = load_dataset("json", data_files=path, split="train")
            print("Loaded queries dataset from jsonl", path)
            self.queries_dataset = self.queries_dataset.map(self.parse_id)
        elif query_or_dataset == "documents":
            self.doc_dataset = load_dataset("json", data_files=path, split="train")
            print("Loaded documents dataset from jsonl", path)
            self.doc_dataset = self.doc_dataset.map(self.parse_id)
        else:
            raise ValueError("query_or_dataset must be either 'queries' or 'documents'")

    @staticmethod
    def parse_id(sample):
        if "chunk_id" in sample:
            doc_id, internal_id = sample["chunk_id"].split("_")
            return {"doc_id": doc_id, "internal_id": internal_id}
        elif "chunk_ids" in sample:
            doc_id, _ = sample["chunk_ids"][0].split("_")
            internal_ids = [id_.split("_")[1] for id_ in sample["chunk_ids"]]
            return {"doc_id": doc_id, "internal_ids": internal_ids, "chunk_id": sample["chunk_ids"]}
        else:
            raise ValueError("chunk_id or chunk_ids not found in sample")

    def get_nested(self, col="chunk") -> tuple[list[list[str]], list[list[str]]]:
        # TODO: verify it's sorted #DONE
        df: pd.DataFrame = self.doc_dataset.to_pandas()
        df.sort_values(by="chunk_id", inplace=True)
        df_by_doc_id = df.groupby("doc_id", sort=True)
        return list(df_by_doc_id[col].apply(list)), list(df_by_doc_id["chunk_id"].apply(list))

    def get_flattened(self, col="chunk") -> tuple[list[str], list[str]]:
        # flatten data
        docs = [f"{self.doc_prompt}{doc}" for doc in self.doc_dataset[col]]
        return docs, self.doc_dataset["chunk_id"]

    def get_queries(self) -> tuple[list[str], list[str]]:
        queries = [f"{self.query_prompt}{q}" for q in self.queries_dataset["query"]]
        return queries, self.queries_dataset["chunk_id"]

    def get_nested_queries(self) -> tuple[list[list[str]], list[list[str]]]:
        df: pd.DataFrame = self.queries_dataset.to_pandas()
        df_by_doc_id = df.groupby("doc_id", sort=True)
        return list(df_by_doc_id["query"].apply(list)), list(df_by_doc_id["chunk_id"].apply(list))
    
    def get_chunks_with_context(self, col="chunk", context_col="context_chunks_ids") -> Generator[tuple[str, str, str], None, None]:
        """Return the chunk and its context, concatenated. Generator"""
        for chunk_id, chunk, context_ids in zip(self.doc_dataset['chunk_id'], self.doc_dataset[col], self.doc_dataset[context_col]):
            if context_ids:
                # context = " ".join([self.doc_dataset.loc[self.doc_dataset["chunk_id"] == cid, col].values[0] for cid in context_ids])  # ty:ignore[not-subscriptable]
                context = " \n ".join(self.doc_dataset.filter(lambda x: x["chunk_id"] in context_ids)[col])
                # print("Context chunks:", context)
                # print("Chunk with context:", f"{self.doc_prompt}{chunk} {context}")

                yield chunk_id, chunk, context

