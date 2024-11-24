import streamlit as st
from dotenv import load_dotenv
import os
import getpass

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
# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 5rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Load the local image file
cur_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(cur_dir, "aimate_symbol_cropped.png")
st.image(logo_path, width=100)
st.header("Умный поиск по документации AimateDocs")

# Create two columns
# col1, col2 = st.columns([0.57, 0.43], vertical_alignment="center")
#
# with col1:
#     st.header("Умный поиск по документации AimateDocs")
#
# with col2:
#     st.image(logo_path, width=150)

# Enable caching for expensive functions
# bootstrap_caching()

sidebar()

main_col1, main_col2 = st.columns([0.45, 0.55], gap='medium')

with main_col2:
    # st.markdown("#### Задайте вопрос")
    with st.form(key="qa_form", border=False):
        query = st.text_area("Задайте вопрос по загруженным документам", height=94)
        st.session_state.query = query
        submit = st.form_submit_button("Отправить")


with main_col1:
    # st.markdown("#### Загрузите документы")
    uploaded_files = st.file_uploader(
        "Загрузите ваши документы",
        type=["pdf", "docx", "txt", "xlsx", "pptx"],
        help="Сканированные документы пока не поддерживаются.",
        accept_multiple_files=True,
    )

    MAX_LINES = 100

    if len(uploaded_files) > MAX_LINES:
        st.warning(f"Достигнуто максимальное количество файлов. Только первые {MAX_LINES} будут обработаны.")
        multiple_files = uploaded_files[:MAX_LINES]

    with st.expander("Расширенные настройки"):
        return_all_chunks = st.checkbox("Показывать все источники", help="Показывать все фрагменты из ваших документов, которые могли быть использованы для генерации ответа.")

        option = st.selectbox(
            "Выберите тип поиска",
            ["Поиск смысловой информации, выявление ключевых концепций", "Поиск конкретной информации, анализ фактов/данных"],
            placeholder="Выберите тип поиска",
            help="Поиск смысловой информации полезен, когда вам нужно понять, о чем идет речь в документах. Поиск конкретной информации полезен, когда вам нужно найти конкретные ответы на вопросы в документах.",
        )

        model_option = st.selectbox(
            "Модель генерации ответа",
            options=["Большая модель (точнее ответы)", "Маленькая модель (быстрее)"],
            help="Большая модель генерирует более точные и информативные ответы, но работает медленнее. Маленькая модель генерирует менее точные и краткие ответы, но работает быстрее.",
        )

        if model_option == "Большая модель (точнее ответы)":
            model = MODEL_LIST[0]
        elif model_option == "Маленькая модель (быстрее)":
            model = MODEL_LIST[1]
        else:
            model = MODEL_LIST[0]

        if option == "Поиск смысловой информации, выявление ключевых концепций":
            chunk_size_input = 1000
            chunk_overlap_input = 400
            num_chunks = 5
        elif option == "Поиск конкретной информации, анализ фактов/данных":
            chunk_size_input = 500
            chunk_overlap_input = 100
            num_chunks = 10
        else:
            chunk_size_input = 1000
            chunk_overlap_input = 400
            num_chunks = 5

        show_full_doc = False

    if not any(uploaded_file for uploaded_file in uploaded_files):
        if submit:
            st.error("Прежде чем отправить запрос, загрузите документы.")
        st.stop()

    def read_files_func(uploaded_files_var):
        files_local = []
        for file in uploaded_files_var:
            try:
                files_local.append(read_file(file))
            except Exception as e:
                display_file_read_error(e, file_name=file.name)
        return files_local

    files = read_files_func(uploaded_files)

    if sum(len(doc.page_content) for file in files for doc in file.docs) > 1_596_007:
        st.warning("Ваши файлы содержат слишком много текста. Пожалуйста, загрузите файлы поменьше.")
        st.stop()

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
        with st.spinner("Обработка документов... Это может занять некоторое время⏳"):
            try:
                folder_index_local = embed_files(
                    files=files_var,
                    embedding=embedding,
                    vector_store=vector_store,
                    model=EMBED_MODEL,
                )
            except Exception as e:
                st.error(f"Ваш запрос не может быть обработан. Вероятно, загруженные документы слишком большие. Пожалуйста, попробуйте удалить некоторые документы или загрузить другие.")
                print("!!!ВОЗНИКЛА ОШИБКА ПРИ ИНДЕКСАЦИИ!!!\nОШИБКА:", e)
                folder_index_local = None

        if folder_index_local is None:
            st.stop()
        st.session_state.files = [file.id for file in files_var]
        st.session_state.folder_index = folder_index_local
        return folder_index_local

    if [file.id for file in chunked_files] != st.session_state.files:
        print("Creating new folder index")
        folder_index = create_folder_index(chunked_files, EMBEDDING, VECTOR_STORE)
        st.success("Документы успешно обработаны!")
    else:
        print("Using existing folder index")
        folder_index = st.session_state.folder_index
        st.success("Документы успешно обработаны!")

    # folder_index = create_folder_index(chunked_files, EMBEDDING, VECTOR_STORE)

with main_col2:
    if submit:
        if not is_query_valid(query):
            st.stop()

        answer_col, sources_col = st.columns(2)

        llm = get_llm(model=model, temperature=0)

        with st.spinner("Ищем ответ на ваш вопрос в документации..."):
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