# from mteb.evaluation.evaluators.RetrievalEvaluator
from __future__ import annotations

import logging

import pytrec_eval

logger = logging.getLogger(__name__)


class CustomRetrievalEvaluator:
    """
    Wrapper class for the MTEB retrieval evaluator.
    """

    def __init__(self, k_values: list[int] = [1, 5, 10, 50, 100]):
        self.k_values = k_values

    def compute_mteb_metrics(
        self,
        relevant_docs: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        **kwargs,
    ) -> dict[str, float]:
        """
        Compute the MTEB retrieval metrics.
        """
        ndcg, _map, recall, precision = self.evaluate(
            relevant_docs,
            results,
            self.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )

        mrr = self.evaluate_custom(relevant_docs, results, self.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            # **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
        }
        return scores

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        if ignore_identical_ids:
            logger.debug(
                "For evaluation, ``ignore_identical_ids=True`` is set to True, the evaluator will ignore "
                "identical query and document ids."
            )
            # Remove identical ids from results dict
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
        else:
            logger.debug(
                "For evaluation, we DO NOT ignore identical query and document ids (default), please explicitly "
                "set ``ignore_identical_ids=True`` to ignore this."
            )

        all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

        for k in k_values:
            all_ndcgs[f"NDCG@{k}"] = []
            all_aps[f"MAP@{k}"] = []
            all_recalls[f"Recall@{k}"] = []
            all_precisions[f"P@{k}"] = []

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
                all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
                all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
                all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

        ndcg, _map, recall, precision = (
            all_ndcgs.copy(),
            all_aps.copy(),
            all_recalls.copy(),
            all_precisions.copy(),
        )

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
            _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
            recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
            precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)

        return ndcg, _map, recall, precision


    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
        output_type: str = "all",
    ) -> dict[str, float]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            metric_scores = mrr(qrels, results, k_values, output_type)
        metric_scores_avg = {k: sum(v) / len(v) if v else 0.0 for k, v in metric_scores.items()}
        return metric_scores_avg


def mrr(qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k_values: list[int], output_type: str = "all") -> dict[str, list[float]]:
    """
    Compute Mean Reciprocal Rank (MRR) at each k in k_values.
    """
    mrr_scores = {f"MRR@{k}": [] for k in k_values}
    for qid, rels in results.items():
        sorted_pids = sorted(rels.items(), key=lambda x: x[1], reverse=True)
        relevant = set(qrels.get(qid, {}))
        for k in k_values:
            rank = 1
            found = False
            for pid, _ in sorted_pids[:k]:
                if pid in relevant:
                    mrr_scores[f"MRR@{k}"].append(1.0 / rank)
                    found = True
                    break
                rank += 1
            if not found:
                mrr_scores[f"MRR@{k}"].append(0.0)
    return mrr_scores