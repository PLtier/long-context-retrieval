from html import escape
import json
from pathlib import Path

import typer


def render_collapsible(title, content, open=False):
    return f'''<details{' open' if open else ''}><summary><b>{escape(title)}</b></summary><pre>{escape(content)}</pre></details>'''

def render_pred_chunks(pred_chunks):
    html = ['<div class="pred-chunks"><b>Predicted Chunks (by rank):</b>']
    for chunk in sorted(pred_chunks, key=lambda c: c.get('rank', 9999)):
        chunk_id = chunk.get('pred_chunk_id', 'N/A')
        score = chunk.get('pred_chunk_score', 'N/A')
        text = chunk.get('pred_chunk_text', '')
        html.append(render_collapsible(f"Rank {chunk.get('rank', '?')}: {chunk_id} (score: {score})", text, open=False))
    html.append('</div>')
    return '\n'.join(html)

def render_record(record):
    html = ['<div class="record">']
    html.append(f'<div><b>Query:</b> {escape(record.get("query", ""))}</div>')
    gt_id = record.get('gt_chunk_id', 'N/A')
    gt_text = record.get('gt_chunk_text', '')
    html.append(render_collapsible(f"Ground Truth Chunk: {gt_id}", gt_text, open=True))
    pred_chunks = record.get('pred_chunks', [])
    html.append(render_pred_chunks(pred_chunks))
    html.append('</div><hr>')
    return '\n'.join(html)

def render_html(records, title):
    html = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        f'<title>{escape(title)}</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 2em; }',
        '.record { margin-bottom: 2em; padding: 1em; border: 1px solid #ccc; border-radius: 8px; background: #fafaff; }',
        'details { margin: 0.5em 0; }',
        'pre { white-space: pre-wrap; word-break: break-word; background: #f4f4f4; padding: 0.5em; border-radius: 4px; }',
        'hr { border: none; border-top: 1px solid #eee; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h1>{escape(title)}</h1>'
    ]
    for rec in records:
        html.append(render_record(rec))
    html.append('</body></html>')
    return '\n'.join(html)

app = typer.Typer()

@app.command()
def main(
    input: str = typer.Argument(..., help="Input .jsonl file (results format)"),
    output: str = typer.Option(None, "-o", "--output", help="Output HTML file (default: input.html)"),
):
    """Visualise results JSONL file as HTML: query, ground truth, predicted chunks."""
    input_path = Path(input)
    if output:
        output_path = Path(output)
    else:
        output_path = input_path.parent / (input_path.stem + '.html')
    records = []
    with input_path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                typer.echo(f"[Line {i}] Skipping malformed line: {e}", err=True)
    html = render_html(records, title=f"Visualisation: {input_path.name}")
    output_path.write_text(html)
    typer.echo(f"Wrote {len(records)} records to {output_path}")

if __name__ == "__main__":
    app()
