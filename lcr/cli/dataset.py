import asyncio
import json
import os
from pathlib import Path

from bs4 import BeautifulSoup
import datasets
from loguru import logger
from tqdm import tqdm
import typer

from lcr.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from lcr.utils import DATASETS, contextualise_datasets_async

app = typer.Typer()


@app.command()
def test(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------

@app.command()
def chunk_document(
    input_dir: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR, # this should be .jsonl file
):
    logger.info("Chunking documents...")

    doc_names = sorted(os.listdir(input_dir))
    chunks = []
    for doc_name in tqdm(doc_names, total=len(doc_names)):
        doc_path = input_dir / doc_name
        doc_chunks = []
        if not doc_path.is_file() or not doc_name.endswith('.html'):
            continue
        doc_name = doc_name.split('.')[0]
        with open(doc_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            # Add first h1 in body as the first chunk for the document
            body = soup.body
            first_h1 = body.find('h1') if body else None
            if first_h1:
                doc_chunks.append({
                    'chunk_id': f"{doc_name}_0",
                    'chunk': first_h1.get_text(strip=True)
                })
            xtext_nodes = soup.select('[data-template="xText"]')
            for node in xtext_nodes:
                chunk_text = node.get_text(strip=True)
                chunk_text = ' '.join(chunk_text.replace('\n', ' ').split())  # Normalize whitespace and \n
                parent = node.parent
                h3_text = None
                # Find h3 sibling of the parent
                for sibling in parent.previous_siblings:
                    if getattr(sibling, 'name', None) == 'h3':
                        h3_text = sibling.get_text(strip=True)
                        break
                # Concatenate h3 text if found
                if h3_text:
                    full_chunk = h3_text + " " + chunk_text
                    if h3_text == '1.':
                        grandgrandparent = parent.parent.parent
                        if grandgrandparent:
                            par_h3_text = None
                            for sibling in grandgrandparent.previous_siblings:
                                if getattr(sibling, 'name', None) == 'h3':
                                    par_h3_text = sibling.get_text(strip=True)
                                    break
                            if par_h3_text:
                                full_chunk = par_h3_text + ' ' + full_chunk
                else:
                    full_chunk = chunk_text
                doc_chunks.append({
                    'chunk_id': f"{doc_name}_{len(doc_chunks)}",
                    'chunk': full_chunk
                })
        chunks.extend(doc_chunks)
    # Save chunks as JSONL
    output_path = Path(output_path) # .jsonl file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # save it as jsonl for a lookup
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for chunk in chunks:
            out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    # also, save it as in hf datasets format:
    dataset = datasets.Dataset.from_list(chunks)
    dataset.save_to_disk(output_path.with_suffix('')) # save without .jsonl suffix for

    logger.success(f"Chunking complete. {len(chunks)} chunks written to {output_path}")

@app.command()
def contextualise_datasets(
    contextualisation_model: str = "",
    datasets: list[str] = list(DATASETS.keys()),
    data_base_path: str = "",
    save_dir: str = "./contextualised_datasets",
):
    asyncio.run(contextualise_datasets_async(
        contextualisation_model, datasets, data_base_path, save_dir
    ))

@app.command()
def queries_to_dataset(path: Path, output_path: Path):
    """Converting from jsonl file to a hf dataset"""
    logger.info(f"Converting queries from {path} to dataset at {output_path}")
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    dataset = datasets.Dataset.from_list(queries)
    dataset.save_to_disk(output_path)
    logger.success(f"Queries converted and saved to {output_path}")
    


if __name__ == "__main__":
    app()
