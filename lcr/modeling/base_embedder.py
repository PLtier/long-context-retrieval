import pandas as pd
import torch

from lcr.eval_utils import CustomRetrievalEvaluator
from lcr.formatter import (
    DataFormatter,
)


class Embedder:
    def __init__(
        self,
        is_contextual_model: bool = False,
        use_nested_queries: bool = False,
        device: str = "cpu",
    ):
        self.is_contextual_model = is_contextual_model
        self.evaluator = CustomRetrievalEvaluator()
        self.use_nested_queries = use_nested_queries
        self.merge_embeddings = False
        self._device = device

    def embed_queries(self, queries):
        raise NotImplementedError

    def embed_documents(self, documents):
        raise NotImplementedError

    def _convert_to_tensor(self, embeddings):
        """
        Convert embeddings to a torch tensor if they are not already.
        """
        if isinstance(embeddings, list):
            if isinstance(embeddings[0], torch.Tensor):
                embeddings = torch.stack(embeddings, dim=0)
            else:
                embeddings = torch.tensor(embeddings)
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        embeddings = embeddings.to(self._device)
        return embeddings

    def process_queries(self, data_formatter: DataFormatter):
        if self.use_nested_queries:
            queries, document_ids = data_formatter.get_nested_queries()
            document_ids = [id_ for nested_ids in document_ids for id_ in nested_ids]
        else:
            queries, document_ids = data_formatter.get_queries()

        query_embeddings = self.embed_queries(queries)

        return query_embeddings, document_ids

    def _has_formatter_beir_format(self, data_formatter: DataFormatter):
        # return isinstance(data_formatter, BEIRDataFormatter) or isinstance(data_formatter, LongEmbedDataFormatter)
        return False

    def process_documents(self, data_formatter: DataFormatter, col="chunk"):
        if self.merge_embeddings:
            raise NotImplementedError

        else:
            if self.is_contextual_model and not self._has_formatter_beir_format(data_formatter):
                documents, document_ids = data_formatter.get_nested(col=col)
                # embed documents in contextual models receive
                # a list of list of documents and should return embeddings in the same shape
                doc_embeddings = self.embed_documents(documents)
                # flatten
                document_ids = [id_ for nested_ids in document_ids for id_ in nested_ids]
                doc_embeddings = [embed_ for nested_embeds in doc_embeddings for embed_ in nested_embeds]
            elif self.is_contextual_model and self._has_formatter_beir_format(data_formatter):
                documents, document_ids = data_formatter.get_flattened()
                doc_embeddings = self.embed_documents([[doc] for doc in documents])
                doc_embeddings = [embed_ for nested_embeds in doc_embeddings for embed_ in nested_embeds]
            else:
                documents, document_ids = (
                    data_formatter.get_flattened()
                    if self._has_formatter_beir_format(data_formatter)
                    else data_formatter.get_flattened(col=col)
                )
                doc_embeddings = self.embed_documents(documents)

        # make into a contiguous tensor, and map position to document_ids
        return doc_embeddings, document_ids

    def get_similarities(self, query_embeddings, doc_embeddings) -> torch.Tensor:
        # convert to torch tensors and compute similarity with dot product
        scores = torch.mm(query_embeddings, doc_embeddings.t())
        return scores.to("cpu")
    
    def get_results(self, scores, all_document_ids) -> dict[str, dict[str, float]]:
        results = {}
        for idx, scores_per_query in enumerate(scores):
            results[str(idx)] = {str(doc_id): score.item() for doc_id, score in zip(all_document_ids, scores_per_query)}
        return results

    def get_metrics(self, scores, all_document_ids, label_documents_id, **kwargs):
        # scores are a list of list of scores (or 2D tensor)
        # label_document_ids are a list of document ids corresponding to the true label.
        # it can also be a list of list of document ids if there are multiple labels per query
        # all_document_ids are a list of all document ids in the same order as the scores

        assert scores.shape[1] == len(all_document_ids)
        assert scores.shape[0] == len(label_documents_id)

        relevant_docs = {}
        if not isinstance(label_documents_id[0], list):
            assert set(label_documents_id).issubset(set(all_document_ids))

            for idx, label in enumerate(label_documents_id):
                relevant_docs[str(idx)] = {label: 1}
        else:
            flatten_labels = [label for nested_labels in label_documents_id for label in nested_labels]
            assert set(flatten_labels).issubset(set(all_document_ids))

            for idx, labels in enumerate(label_documents_id):
                relevant_docs[str(idx)] = {label: 1 for label in labels}

        results = self.get_results(scores, all_document_ids)

        metrics: dict[str, float] = self.evaluator.compute_mteb_metrics(relevant_docs, results)
            # Add per-query results for downstream CSV extraction
        # metrics["results"] = results

        if kwargs.get("max_sim_over_docs", False):
            # relevant docs (max similarity over all docs with same prefix)
            new_relevant_docs = {}
            for key, doc in relevant_docs.items():
                new_doc = {k.split("_")[0]: v for k, v in doc.items()}
                new_relevant_docs[key] = new_doc

            results_df = pd.DataFrame(results)
            results_df["doc_id"] = results_df.index.map(lambda x: x.split("_")[0])
            results_df = results_df.groupby("doc_id").max()
            new_results = results_df.to_dict()
            metrics_tmp = self.evaluator.compute_mteb_metrics(new_relevant_docs, new_results)
            metrics["ndcg_at_10_maxsim"] = metrics_tmp["ndcg_at_10"]

        return metrics

    def compute_results(
        self, data_formatter: DataFormatter, **kwargs
    ) -> tuple[dict[str, dict[str, float]], list[str], dict[str, float]]:
        import time

        t0 = time.time()
        queries_embeddings, label_ids = self.process_queries(data_formatter)
        queries_embeddings = self._convert_to_tensor(queries_embeddings)
        query_embedding_time = time.time() - t0

        t0 = time.time()
        documents_embeddings, all_doc_ids = self.process_documents(data_formatter)
        documents_embeddings = self._convert_to_tensor(documents_embeddings)
        doc_embedding_time = time.time() - t0

        if kwargs.get("mean_over_docs", False):
            # average doc embeddings between themselves if they have the same all_doc_ids prefix
            all_doc_ids = [doc_id.split("_")[0] for doc_id in all_doc_ids]
            unique_doc_ids = set(all_doc_ids)
            new_documents_embeddings = []
            new_all_doc_ids = []
            for doc_id in unique_doc_ids:
                idxs = [i for i, x in enumerate(all_doc_ids) if x == doc_id]
                new_documents_embeddings.append(torch.mean(torch.stack([documents_embeddings[i] for i in idxs]), dim=0))
                new_all_doc_ids.append(doc_id)
            documents_embeddings = torch.stack(new_documents_embeddings)
            all_doc_ids = new_all_doc_ids
            label_ids = [doc_id.split("_")[0] for doc_id in label_ids]

        t0 = time.time()
        scores = self.get_similarities(queries_embeddings, documents_embeddings)
        scores = scores.numpy()
        similarity_time = time.time() - t0

        results = self.get_results(scores, all_doc_ids)
        metrics = self.get_metrics(scores, all_doc_ids, label_ids, **kwargs)
        metrics["query_embedding_time"] = query_embedding_time
        metrics["doc_embedding_time"] = doc_embedding_time
        metrics["similarity_time"] = similarity_time

        return results, label_ids, metrics
