import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import torch
import typer

from lcr.formatter import DataFormatter
from lcr.modeling.sentence_transformer_embedder import SentenceTransformerEmbedder
from lcr.utils import DATASETS

app = typer.Typer()




DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@app.command()
def eval_model(
    model_path: str = "",
    documents_base_dir: str = "",
    query_base_dir: str = "",
    save_dir: str = "",
    datasets: list[str] = list(DATASETS.keys()),
    use_prefix: bool = False,
):
    if model_path == "" or query_base_dir == "" or save_dir == "" or documents_base_dir == "":
        print("Please provide all required arguments: --model-path, --query-base-dir, --documents-base-dir, and --save-dir")
        return
    model_name = model_path.split("/")[-1]
    embedder = SentenceTransformerEmbedder(SentenceTransformer(model_path, trust_remote_code=True, device=DEVICE), batch_size=128, device=DEVICE, add_prefix=use_prefix)



    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"Dataset {dataset} not found in DATASETS. Skipping.")
            continue

        # just in case
        path = DATASETS[dataset].get("path", dataset)
        split = DATASETS[dataset].get("split", "train")
        is_query_local = DATASETS[dataset].get("is_query_local", False)
        is_docs_local = DATASETS[dataset].get("is_docs_local", False)


        ds_formatter = DataFormatter()
        ds_formatter.load_documents(f"{documents_base_dir}/{path}", is_local=is_docs_local)
        ds_formatter.load_queries(f"{query_base_dir}/{path}", is_local=is_query_local, split=split)
        ds_formatter.query_prompt = ""; ds_formatter.doc_prompt =""







        # formatters[dataset] = ds_formatter

        preds, labels, metrics = embedder.compute_results(ds_formatter)
        labels = list(labels)

        output_path = f"{save_dir}/{dataset}_{model_name}_results.json"
        # with open(output_path, "w") as f:
            # json.dump({"preds": preds, "labels": labels, "metrics": metrics}, f)
        print(f"Saved results for {dataset} to {output_path}")
        print(metrics)


        top_n = 5
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
                'pred_chunks': []
            }
            if not pred_scores:
                p_chunk_id = ''
                p_chunk = ''
            else:
                top_chunks = sorted(pred_scores, key=pred_scores.get, reverse=True)[:top_n]
                
                # p_chunk_id = max(pred_scores, key=pred_scores.get)
                # p_chunk = chunk_id_to_text.get(p_chunk_id, '')
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

        jsonl_filename = f"output_{dataset}_.jsonl"
        jsonl_path = f"{save_dir}/{jsonl_filename}"
        # create the save dir it if doesn't exist
        Path(save_dir).mkdir(parents=False, exist_ok=True)
        with open(jsonl_path, 'w') as jsonlfile:
            for row in jsonl_rows:
                jsonlfile.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"JSON Lines output saved to {jsonl_path}!")




if __name__ == "__main__":
    app()

