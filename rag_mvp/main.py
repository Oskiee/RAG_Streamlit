import streamlit as st
from dotenv import load_dotenv
import os
import getpass
import hashlib
import pickle
import atexit

from components.sidebar import sidebar
from ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    display_file_read_error,
)
from core.caching import bootstrap_caching
from core.parsing import read_file
from core.chunking import chunk_file
from core.embedding import embed_files
from core.qa import query_folder
from core.utils import get_llm

load_dotenv()

EMBEDDING = "mistral"
EMBED_MODEL = "mistral-embed"
VECTOR_STORE = "faiss"
MODEL_LIST = ["mistral-large-latest", "mistral-small-latest"]

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

st.set_page_config(page_title="RAG_MVP", page_icon="🔍", layout="wide")
st.header("🔍Умный поиск по документации AimateDocs")

# Enable caching for expensive functions
# bootstrap_caching()

sidebar()

uploaded_files = st.file_uploader(
    "Загрузите pdf, docx, или txt файл",
    type=["pdf", "docx", "txt"],
    help="Сканированные документы пока не поддерживаются.",
    accept_multiple_files=True,
)

MAX_LINES = 100

if len(uploaded_files) > MAX_LINES:
    st.warning(f"Достигнуто максимальное количество файлов. Только первые {MAX_LINES} будут обработаны.")
    multiple_files = uploaded_files[:MAX_LINES]

model: str = st.selectbox("Модель", options=MODEL_LIST)  # type: ignore

with st.expander("Расширенные опции"):
    return_all_chunks = st.checkbox("Показывать все текстовые блоки, извлеченные векторным поиском")
    chunk_size_input = st.number_input(
        "Размер текстового блока (в символах)",
        min_value=100,
        max_value=2000,
        help="Значение должно быть в пределах от 100 до 2000 символов",
        value=1000,
    )
    chunk_overlap_input = st.number_input(
        "Наложение текстовых блоков (в символах)",
        min_value=0,
        max_value=1500,
        help="Значение должно быть в пределах от 0 до 1500 символов",
        value=400,
    )
    num_chunks = st.number_input(
        "Максимальное количество текстовых блоков для поиска",
        min_value=1,
        max_value=30,
        value=5,
        help="Значение должно быть в пределах от 1 до 30",
    )
    chunk_size_input = int(chunk_size_input)
    chunk_overlap_input = int(chunk_overlap_input)
    num_chunks = int(num_chunks)
    show_full_doc = False

if chunk_size_input <= chunk_overlap_input:
    st.error("Размер блока не может быть меньше наложения блоков")
    st.stop()

if not any(uploaded_file for uploaded_file in uploaded_files):
    st.stop()

if not (100 <= chunk_size_input <= 2000 and 0 <= chunk_overlap_input <= 1500):
    st.stop()


# @st.cache_data(show_spinner=False, ttl="8h")
def read_files_func(uploaded_files_var):
    files_local = []
    for file in uploaded_files_var:
        try:
            files_local.append(read_file(file))
        except Exception as e:
            display_file_read_error(e, file_name=file.name)
    return files_local


files = read_files_func(uploaded_files)


# @st.cache_data(show_spinner=False, ttl="8h")
def chunk_files_func(files_var, chunk_size, chunk_overlap):
    chunked_files_local = []
    for file in files_var:
        chunked_file = chunk_file(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_files_local.append(chunked_file)
    return chunked_files_local


chunked_files = chunk_files_func(files, chunk_size=chunk_size_input, chunk_overlap=chunk_overlap_input)

if not any(is_file_valid(chunked_file) for chunked_file in chunked_files):
    st.stop()

if 'files' not in st.session_state:
    st.session_state.files = []

if 'folder_index' not in st.session_state:
    st.session_state.folder_index = None

# @st.cache_resource(show_spinner=False, ttl="8h")
def create_folder_index(files_var, embedding, vector_store):
    with st.spinner("Индексация документов... Это может занять некоторое время⏳"):
        folder_index_local = embed_files(
            files=files_var,
            embedding=embedding,
            vector_store=vector_store,
            model=EMBED_MODEL,
        )

    # save ids of files in session
    st.session_state.files = [file.id for file in files_var]
    st.session_state.folder_index = folder_index_local
    return folder_index_local


print(st.session_state.files)
print([file.id for file in chunked_files])
if [file.id for file in chunked_files] != st.session_state.files:
    print("Creating new folder index")
    folder_index = create_folder_index(chunked_files, EMBEDDING, VECTOR_STORE)
else:
    print("Using existing folder index")
    folder_index = st.session_state.folder_index

# folder_index = create_folder_index(chunked_files, EMBEDDING, VECTOR_STORE)

with st.form(key="qa_form"):
    query = st.text_area("Задайте вопрос по документу")
    submit = st.form_submit_button("Отправить")

# if show_full_doc:
#     with st.expander("Document"):
#         st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)
#
# def get_query_folder(folder_index, query, return_all_chunks, llm, num_chunks):
#

if submit:
    if not is_query_valid(query):
        st.stop()

    answer_col, sources_col = st.columns(2)

    llm = get_llm(model=model, temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
        num_sources=num_chunks,
    )

    with answer_col:
        st.markdown("#### Ответ")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Источники")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
