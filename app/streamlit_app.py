"""Streamlit веб-чат по документу.

Запуск:  streamlit run app/streamlit_app.py

Возможности:
  * загрузка своего PDF/TXT или демо-корпус SQuAD v2;
  * чат с показом ответа, уверенности и ЦИТАТЫ источника;
  * явная индикация случая «в документе нет ответа».
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Позволяем запускать из корня репозитория без установки пакета.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shkerebert.config import default_config
from shkerebert.ingest import load_document, load_squad_documents
from shkerebert.pipeline import RAGPipeline

st.set_page_config(page_title="ShkereBERT — чат по документу", page_icon="📄")


@st.cache_resource(show_spinner=True)
def build_pipeline_squad(n: int):
    docs = load_squad_documents(split="validation", max_contexts=n)
    return RAGPipeline.build(docs, default_config(), show_progress=False)


@st.cache_resource(show_spinner=True)
def build_pipeline_file(path: str):
    docs = [load_document(path)]
    return RAGPipeline.build(docs, default_config(), show_progress=False)


st.title("📄 ShkereBERT")
st.caption("RAG-чат-бот по документу с цитированием источника и обработкой «нет ответа».")

with st.sidebar:
    st.header("Источник документов")
    mode = st.radio("Что используем?", ["Демо: SQuAD v2", "Свой файл (PDF/TXT)"])
    pipe = None
    if mode == "Демо: SQuAD v2":
        n = st.slider("Контекстов SQuAD", 50, 1000, 200, step=50)
        if st.button("Загрузить корпус"):
            st.session_state["pipe_key"] = ("squad", n)
    else:
        up = st.file_uploader("Загрузите PDF или TXT", type=["pdf", "txt"])
        if up is not None:
            save_path = Path("data") / f"upload_{up.name}"
            save_path.parent.mkdir(exist_ok=True)
            save_path.write_bytes(up.getbuffer())
            st.session_state["pipe_key"] = ("file", str(save_path))

key = st.session_state.get("pipe_key")
if key:
    if key[0] == "squad":
        pipe = build_pipeline_squad(key[1])
    else:
        pipe = build_pipeline_file(key[1])

if pipe is None:
    st.info("Выберите источник документов слева и нажмите «Загрузить».")
    st.stop()

question = st.text_input("Ваш вопрос", placeholder="Например: What is the capital of ...?")
if question:
    ans = pipe.answer(question)
    if ans.is_answerable:
        st.success(f"**Ответ:** {ans.answer}")
    else:
        st.warning("**В документе нет ответа на этот вопрос.**")
    st.caption(f"confidence = {ans.confidence:.2f} · {ans.reason}")

    st.markdown("#### Источники")
    for s in ans.sources[:5]:
        title = f"{'⭐ ' if s.is_answer_source else ''}[{s.chunk_id}] score={s.retrieval_score:.3f}"
        with st.expander(title, expanded=s.is_answer_source):
            st.write(s.text)
