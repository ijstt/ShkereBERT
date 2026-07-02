"""Интерактивный CLI-чат по документу.

Примеры:
  python -m shkerebert.cli chat --file path/to/doc.pdf
  python -m shkerebert.cli chat --squad-n 300           # демо на подвыборке SQuAD
  python -m shkerebert.cli ask --file doc.txt "What is ...?"
"""

from __future__ import annotations

import typer

from .config import default_config
from .ingest import load_document, load_squad_documents
from .pipeline import Answer, RAGPipeline

app = typer.Typer(add_completion=False, help="ShkereBERT — чат-бот по документу (RAG).")


def _build(file: str | None, squad_n: int) -> RAGPipeline:
    cfg = default_config()
    if file:
        docs = [load_document(file)]
        typer.echo(f"Загружен документ: {file} ({len(docs[0].text)} символов)")
    else:
        docs = load_squad_documents(split="validation", max_contexts=squad_n)
        typer.echo(f"Загружено {len(docs)} контекстов SQuAD v2")
    typer.echo("Строю индекс (первый раз считаются эмбеддинги)...")
    return RAGPipeline.build(docs, cfg)


def _print_answer(ans: Answer, show_sources: bool = True) -> None:
    typer.echo("")
    if ans.is_answerable:
        typer.secho(f"Ответ: {ans.answer}", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho("Ответ: В документе нет ответа на этот вопрос.",
                    fg=typer.colors.YELLOW, bold=True)
    typer.echo(f"(confidence={ans.confidence:.2f}; {ans.reason})")
    if show_sources and ans.sources:
        typer.echo("Источники:")
        for s in ans.sources[:3]:
            mark = "★" if s.is_answer_source else " "
            snippet = s.text[:160].replace("\n", " ")
            typer.echo(f"  {mark} [{s.chunk_id}] score={s.retrieval_score:.3f}: {snippet}...")


@app.command()
def chat(
    file: str = typer.Option(None, help="Путь к .txt/.pdf документу"),
    squad_n: int = typer.Option(200, help="Сколько контекстов SQuAD взять, если нет --file"),
):
    """Интерактивный чат: задавайте вопросы, 'exit' для выхода."""
    pipe = _build(file, squad_n)
    typer.secho("\nГотово! Задавайте вопросы (exit/quit — выход).\n", fg=typer.colors.CYAN)
    while True:
        try:
            q = typer.prompt("Вопрос")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"exit", "quit", ""}:
            break
        _print_answer(pipe.answer(q))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Вопрос"),
    file: str = typer.Option(None, help="Путь к .txt/.pdf документу"),
    squad_n: int = typer.Option(200, help="Контекстов SQuAD, если нет --file"),
):
    """Одиночный вопрос (для скриптов)."""
    pipe = _build(file, squad_n)
    _print_answer(pipe.answer(question))


if __name__ == "__main__":
    app()
