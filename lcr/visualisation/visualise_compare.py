from html import escape
import json
from pathlib import Path
from typing import Optional

import typer


def rank_badge(rank):
    if rank == 1:
        bg = '#2ecc71'
    elif rank is not None and rank <= 5:
        bg = '#f39c12'
    else:
        bg = '#e74c3c'
    label = str(rank) if rank is not None else 'N/A'
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold;">Rank {label}</span>'


def render_collapsible(title, content, open=False):
    return f'<details{" open" if open else ""}><summary><b>{escape(title)}</b></summary><pre>{escape(content)}</pre></details>'


def render_pred_chunks(pred_chunks):
    html = ['<div class="pred-chunks"><b>Predicted Chunks (by rank):</b>']
    for chunk in sorted(pred_chunks, key=lambda c: c.get('rank', 9999)):
        chunk_id = chunk.get('pred_chunk_id', 'N/A')
        score = chunk.get('pred_chunk_score', 'N/A')
        text = chunk.get('pred_chunk_text', '')
        html.append(render_collapsible(f"Rank {chunk.get('rank', '?')}: {chunk_id} (score: {score})", text))
    html.append('</div>')
    return '\n'.join(html)


def render_query_cell(q_rec):
    if q_rec is None:
        return '<div class="cell missing">No query record</div>'
    html = ['<div class="cell">']
    html.append(f'<div class="query-text">{escape(q_rec.get("query", ""))}</div>')
    html.append(f'<div><b>chunk_id:</b> {escape(q_rec.get("chunk_id", "N/A"))}</div>')
    ids = q_rec.get('utilized_context_chunk_ids', [])
    ids_str = ', '.join(ids) if isinstance(ids, list) else str(ids)
    html.append(f'<div><b>utilized_context_chunk_ids:</b> {escape(ids_str)}</div>')
    if q_rec.get('chunk'):
        html.append(render_collapsible('chunk', q_rec['chunk']))
    if q_rec.get('context_chunks'):
        html.append(render_collapsible('context_chunks', q_rec['context_chunks']))
    if q_rec.get('impl_context_chunks'):
        html.append(render_collapsible('impl_context_chunks', q_rec['impl_context_chunks']))
    html.append('</div>')
    return '\n'.join(html)


def render_results_cell(r_rec):
    if r_rec is None:
        return '<div class="cell missing">No result record</div>'
    html = ['<div class="cell">']
    gt_rank = r_rec.get('gt_chunk_rank')
    html.append(f'<div>{rank_badge(gt_rank)}</div>')
    gt_id = r_rec.get('gt_chunk_id', 'N/A')
    gt_text = r_rec.get('gt_chunk_text', '')
    html.append(render_collapsible(f'GT: {gt_id}', gt_text, open=True))
    pred_chunks = r_rec.get('pred_chunks', [])
    if pred_chunks:
        html.append(render_pred_chunks(pred_chunks))
    html.append('</div>')
    return '\n'.join(html)


def render_row(query, q_rec, ra_rec, rb_rec):
    return (
        f'<div class="row">'
        f'{render_query_cell(q_rec)}'
        f'{render_results_cell(ra_rec)}'
        f'{render_results_cell(rb_rec)}'
        f'</div>'
    )


def render_html(rows_html, label_a, label_b, title):
    return '\n'.join([
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        f'<title>{escape(title)}</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 0; padding: 1em; }',
        '.header { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;'
        ' position: sticky; top: 0; background: #fff; border-bottom: 2px solid #aaa;'
        ' padding: 0.5em 0; z-index: 10; font-size: 1.1em; font-weight: bold; text-align: center; }',
        '.row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;'
        ' margin-bottom: 1em; border-bottom: 1px solid #ddd; padding-bottom: 1em; }',
        '.cell { padding: 0.75em; border: 1px solid #ccc; border-radius: 6px; background: #fafaff;'
        ' min-width: 0; overflow: hidden; }',
        '.cell.missing { color: #999; font-style: italic; }',
        '.query-text { font-weight: bold; margin-bottom: 0.5em; }',
        'details { margin: 0.4em 0; }',
        'pre { white-space: pre-wrap; word-break: break-word; background: #f4f4f4;'
        ' padding: 0.5em; border-radius: 4px; max-height: 300px; overflow-y: auto; }',
        '.pred-chunks { margin-top: 0.5em; }',
        'h1 { margin-top: 0; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h1>{escape(title)}</h1>',
        f'<div class="header"><div>Query</div><div>{escape(label_a)}</div><div>{escape(label_b)}</div></div>',
        *rows_html,
        '</body></html>',
    ])


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                typer.echo(f"[{path.name} line {i}] Skipping malformed line: {e}", err=True)
    return records


app = typer.Typer()


@app.command()
def main(
    queries: str = typer.Argument(..., help="Path to contextual_queries.jsonl"),
    results_a: str = typer.Argument(..., help="Path to first results JSONL (e.g. bge-m3)"),
    results_b: str = typer.Argument(..., help="Path to second results JSONL (e.g. bm25)"),
    label_a: str = typer.Option("BGE-M3", "--label-a", help="Label for results A column"),
    label_b: str = typer.Option("BM25", "--label-b", help="Label for results B column"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output HTML file"),
    ignore_mismatch: bool = typer.Option(False, "--ignore-mismatch", help="Warn instead of abort on query set mismatch"),
):
    """Visualise BGE-M3 vs BM25 results alongside query context in a three-column layout."""
    queries_path = Path(queries)
    ra_path = Path(results_a)
    rb_path = Path(results_b)
    output_path = Path(output) if output else ra_path.parent / (ra_path.stem + '_compare.html')

    q_records = load_jsonl(queries_path)
    ra_records = load_jsonl(ra_path)
    rb_records = load_jsonl(rb_path)

    q_by_query = {r['query']: r for r in q_records if 'query' in r}
    ra_by_query = {r['query']: r for r in ra_records if 'query' in r}
    rb_by_query = {r['query']: r for r in rb_records if 'query' in r}

    all_keys = set(q_by_query) | set(ra_by_query) | set(rb_by_query)
    common_keys = set(q_by_query) & set(ra_by_query) & set(rb_by_query)
    missing = all_keys - common_keys

    if missing:
        msg = f"Query mismatch: {len(missing)} queries not present in all three files."
        if not ignore_mismatch:
            typer.echo(f"Error: {msg}", err=True)
            for q in sorted(missing):
                in_files = []
                if q in q_by_query: in_files.append('queries')
                if q in ra_by_query: in_files.append(label_a)
                if q in rb_by_query: in_files.append(label_b)
                typer.echo(f"  only in [{', '.join(in_files)}]: {q[:80]!r}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Warning: {msg} Proceeding with intersection.", err=True)
        for q in sorted(missing):
            in_files = []
            if q in q_by_query: in_files.append('queries')
            if q in ra_by_query: in_files.append(label_a)
            if q in rb_by_query: in_files.append(label_b)
            typer.echo(f"  only in [{', '.join(in_files)}]: {q[:80]!r}", err=True)

    def sort_key(q):
        rec = q_by_query.get(q)
        if rec:
            return (rec.get('chunk_id', ''), q)
        return ('', q)

    sorted_queries = sorted(common_keys, key=sort_key)

    rows_html = [
        render_row(q, q_by_query.get(q), ra_by_query.get(q), rb_by_query.get(q))
        for q in sorted_queries
    ]

    html = render_html(rows_html, label_a, label_b, title=f"{label_a} vs {label_b}: {queries_path.name}")
    output_path.write_text(html, encoding='utf-8')
    typer.echo(f"Wrote {len(sorted_queries)} rows to {output_path}")


if __name__ == '__main__':
    app()
