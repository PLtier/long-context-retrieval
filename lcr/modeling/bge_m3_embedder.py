import time

from FlagEmbedding import BGEM3FlagModel
import numpy as np
from seaborn.external.husl import m

from lcr.modeling.base_embedder import Embedder


class BGEM3Embedder(Embedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 16,
        encoding_type: str = "dense",  # "dense" | "sparse"
        device: str = "cpu",
    ):
        super().__init__(is_contextual_model=False, device=device)
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device, batch_size=batch_size)
        self.encoding_type = encoding_type

    def _encode(self, texts):
        return self.model.encode(
            texts,
            return_dense=(self.encoding_type == "dense"),
            return_sparse=(self.encoding_type == "sparse"),
            return_colbert_vecs=False,
            max_length=8192,
        )

    def embed_queries(self, queries):
        out = self._encode(queries)
        return out["dense_vecs"] if self.encoding_type == "dense" else out["lexical_weights"]

    def embed_documents(self, documents):
        out = self._encode(documents)
        return out["dense_vecs"] if self.encoding_type == "dense" else out["lexical_weights"]

    def compute_results(self, data_formatter, **kwargs):
        if self.encoding_type == "dense":
            return super().compute_results(data_formatter, **kwargs)

        # Sparse path: embeddings are list[dict[token, weight]].
        # _convert_to_tensor and torch.mm don't apply; compute pairwise
        # lexical matching scores via the model's built-in function.
        t0 = time.time()
        query_embeddings, label_ids = self.process_queries(data_formatter)
        query_embedding_time = time.time() - t0

        t0 = time.time()
        doc_embeddings, all_doc_ids = self.process_documents(data_formatter)
        doc_embedding_time = time.time() - t0

        t0 = time.time()
        scores = np.array([
            [self.model.compute_lexical_matching_score(q, d) for d in doc_embeddings]
            for q in query_embeddings
        ])
        similarity_time = time.time() - t0

        results = self.get_results(scores, all_doc_ids)
        metrics = self.get_metrics(scores, all_doc_ids, label_ids, **kwargs)
        metrics["query_embedding_time"] = query_embedding_time
        metrics["doc_embedding_time"] = doc_embedding_time
        metrics["similarity_time"] = similarity_time
        return results, label_ids, metrics
