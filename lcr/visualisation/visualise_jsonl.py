from html import escape
import json
from pathlib import Path

import typer


def render_collapsible(title, content, open=False):
    return f'''<details{' open' if open else ''}><summary><b>{escape(title)}</b></summary><pre>{escape(content)}</pre></details>'''

def render_record(record, preset):
    html = ['<div class="record">']
    for k, v in record.items():
        # Large fields collapsed by default
        if preset == 'queries' and k in {'context_chunks', 'impl_context_chunks', 'chunk'}:
            html.append(render_collapsible(k, v, open=False))
        elif preset == 'assurance_results' and k in {'passes_assurance'}:
            html.append(render_collapsible(k, v, open=False))
        else:
            html.append(f'<div><b>{escape(k)}:</b> {escape(str(v))}</div>')
    html.append('</div><hr>')
    return '\n'.join(html)

def render_html(records, preset, title):
    html = [
        f'<!DOCTYPE html>',
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
        html.append(render_record(rec, preset))
    html.append('</body></html>')
    return '\n'.join(html)


app = typer.Typer()

@app.command()
def main(
    input: str = typer.Argument(..., help="Input .jsonl file"),
    output: str = typer.Option(None, "-o", "--output", help="Output HTML file (default: input.html)"),
    preset: str = typer.Option(..., help="Preset for known file type", case_sensitive=False),
):
    """Visualise JSONL file as HTML with collapsible large fields."""
    input_path = Path(input)
    output_path = Path(output) if output else input_path.with_suffix('.html')
    preset = preset.lower()
    if preset not in {"queries", "assurance_results"}:
        typer.echo("Error: --preset must be 'queries' or 'assurance_results'", err=True)
        raise typer.Exit(1)
    records = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                typer.echo(f"Skipping line due to error: {e}", err=True)
    title = f"Visualisation of {input_path.name}"
    html = render_html(records, preset, title)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    typer.echo(f"Wrote {output_path}")

if __name__ == '__main__':
    app()
