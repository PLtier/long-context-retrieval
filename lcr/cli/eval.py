import json

from sentence_transformers import SentenceTransformer
import torch
import typer

from lcr.formatter import DataFormatter
from lcr.modeling.bge_m3_embedder import BGEM3Embedder
from lcr.modeling.sentence_transformer_embedder import SentenceTransformerEmbedder

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

app = typer.Typer()


@app.command()
def eval(
    model_path: str = "",
    chunks_path: str = "", # if the docs path is different from the queries path
    queries_path: str = "",
    save_results_dir: str = "",
    use_prefix: bool = False,
    encoder: str = "sentence",  # "sentence" | "flag"
    encoding_type: str = "dense",  # "dense" | "sparse", only applicable if encoder is "flag"
):
    if model_path == "" or queries_path == "" or save_results_dir == "" or chunks_path == "":
        print("Please provide all required arguments: --model-path, --queries-path, --chunks-path, and --save-results-dir")
        return

    model_name = model_path.split("/")[-1]
    if encoder == "flag":
        embedder = BGEM3Embedder(model_name=model_path, batch_size=128, device=DEVICE, encoding_type=encoding_type)
    else:
        embedder = SentenceTransformerEmbedder(SentenceTransformer(model_path, trust_remote_code=True, device=DEVICE), batch_size=128, device=DEVICE, add_prefix=use_prefix)



    ds_formatter = DataFormatter()
    ds_formatter.load_from_jsonl(queries_path, query_or_dataset="queries")
    ds_formatter.load_from_jsonl(chunks_path, query_or_dataset="documents")

    # formatters[dataset] = ds_formatter

    preds, labels, metrics = embedder.compute_results(ds_formatter)
    labels = list(labels)

    output_path = f"{save_results_dir}/{model_name}_results.json"
    # with open(output_path, "w") as f:
        # json.dump({"preds": preds, "labels": labels, "metrics": metrics}, f)
    print(f"Saved results for {model_name} to {output_path}")
    print(metrics)



    top_n = 10
    # save as jsonlines file:
    queries, gt_chunk_ids = ds_formatter.get_queries()
    chunks, chunk_ids = ds_formatter.get_flattened()
    chunk_id_to_text = dict(zip(chunk_ids, chunks))
    jsonl_rows = []
    for idx, query in enumerate(queries):
        gt_id = gt_chunk_ids[idx]
        gt_text = chunk_id_to_text.get(gt_id, '')
        pred_scores = preds.get(str(idx), {})
        entry = {
            'query': query,
            'gt_chunk_text': gt_text,
            'gt_chunk_id': gt_id,
            'gt_chunk_rank': None,
            'pred_chunks': []
        }
        if not pred_scores:
            p_chunk_id = ''
            p_chunk = ''
        else:
            all_ranked = sorted(pred_scores, key=pred_scores.get, reverse=True)
            gt_rank = all_ranked.index(gt_id) + 1 if gt_id in all_ranked else None
            entry['gt_chunk_rank'] = gt_rank
            top_chunks = all_ranked[:top_n]
            for rank, p_chunk_id in enumerate(top_chunks, start=1):
                p_chunk_text = chunk_id_to_text.get(p_chunk_id, '')
                p_chunk_score = pred_scores[p_chunk_id]
                entry['pred_chunks'].append({
                    'pred_chunk_text': p_chunk_text,
                    'pred_chunk_id': p_chunk_id,
                    'pred_chunk_score': round(p_chunk_score, 3),
                    'rank': rank
                })
            jsonl_rows.append(entry)

    import os
    jsonl_filename = f"results_{model_name}.jsonl"
    jsonl_path = f"{save_results_dir}/{jsonl_filename}"
    os.makedirs(save_results_dir, exist_ok=True)
    with open(jsonl_path, 'w') as jsonlfile:
        for row in jsonl_rows:
            jsonlfile.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"JSON Lines output saved to {jsonl_path}!")


    




if __name__ == "__main__":
    app()
