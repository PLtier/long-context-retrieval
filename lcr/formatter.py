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

    @staticmethod
    def parse_id(sample):
        if "chunk_id" in sample:
            doc_id, internal_id = sample["chunk_id"].split("_")
            return {"doc_id": doc_id, "internal_id": int(internal_id)}
        elif "chunk_ids" in sample:
            doc_id, _ = sample["chunk_ids"][0].split("_")
            internal_ids = [int(id_.split("_")[1]) for id_ in sample["chunk_ids"]]
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


