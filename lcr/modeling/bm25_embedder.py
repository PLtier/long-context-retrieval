import time
from typing import Literal

import bm25s
import spacy

from lcr.formatter import DataFormatter
from lcr.modeling.base_embedder import Embedder


class BM25Embedder(Embedder):
    def __init__(self, spacy_model: Literal["pl_core_news_sm", "da_core_news_sm"]):
        super().__init__()
        self.nlp = spacy.load(spacy_model)
        self.retriever: bm25s.BM25

    def _tokenize(self, text: str) -> list[str]:
        return [t.lemma_.lower() for t in self.nlp(text) if not t.is_stop and not t.is_punct]

    def _tokenize_list(self, texts: list[str]) -> list[list[str]]:
        return [self._tokenize(text) for text in texts]

    def tokenize_queries(self, data_formatter: DataFormatter) -> tuple[list[list[str]], list[str]]:
        queries, gt_chunk_ids = data_formatter.get_queries()
        return self._tokenize_list(queries), gt_chunk_ids

    def build_index(self, data_formatter: DataFormatter, col="") -> list[str]:
        chunks, chunk_ids = data_formatter.get_flattened()
        self.retriever = bm25s.BM25(corpus=chunk_ids)
        self.retriever.index(self._tokenize_list(chunks), )
        return chunk_ids

    def compute_results(
        self, data_formatter: DataFormatter, **kwargs
    ) -> tuple[dict[str, dict[str, float]], list[str], dict[str, float]]:
        t0 = time.time()
        tokenized_queries, gt_chunk_ids = self.tokenize_queries(data_formatter)
        query_embedding_time = time.time() - t0

        t0 = time.time()
        all_chunk_ids = self.build_index(data_formatter)
        doc_embedding_time = time.time() - t0

        k = len(all_chunk_ids)
        t0 = time.time()
        pred_chunks_ids, scores = self.retriever.retrieve(
            tokenized_queries, k=k, sorted=False, return_as="tuple", show_progress=True
        ) # no need to sort here.
        similarity_time = time.time() - t0

        results = {}
        relevant_docs = {}
        for idx, gt_chunk_id in enumerate(gt_chunk_ids):
            results[str(idx)] = {
                chunk_id: float(score) for chunk_id, score in zip(pred_chunks_ids[idx], scores[idx])
            }
            relevant_docs[str(idx)] = {gt_chunk_id: 1}

        metrics: dict[str, float] = self.evaluator.compute_mteb_metrics(relevant_docs, results)
        metrics["query_embedding_time"] = query_embedding_time
        metrics["doc_embedding_time"] = doc_embedding_time
        metrics["similarity_time"] = similarity_time

        return results, gt_chunk_ids, metrics
