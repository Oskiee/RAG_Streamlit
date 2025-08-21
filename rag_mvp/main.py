from streamlit_float import *
import streamlit as st
from dotenv import load_dotenv
import os
import getpass
from tqdm import tqdm

from components.sidebar import sidebar
from ui import (
    is_query_valid,
    is_file_valid,
    display_file_read_error,
)
from core.parsing import read_file
from core.chunking import chunk_file
from core.embedding import embed_files
from core.qa import query_folder
from core.utils import get_llm
from core.config import OPENROUTER_MODEL_NAME


load_dotenv()

EMBEDDING = "mistral"
EMBED_MODEL = "mistral-embed"
VECTOR_STORE = "faiss"
MODEL_LIST = [OPENROUTER_MODEL_NAME, OPENROUTER_MODEL_NAME, OPENROUTER_MODEL_NAME]
MAX_LINES = 100


def create_folder_index(files_var, embedding, vector_store):
    with st.spinner("Обработка документов... Это может занять некоторое время⏳"):
        # try:
            folder_index_local = embed_files(
                files=files_var,
                embedding=embedding,
                vector_store=vector_store,
                model=EMBED_MODEL,
            )
        # except Exception as e:
        #     st.error(
        #         f"Ваш запрос не может быть обработан. Вероятно, документ поврежден или имеет нестандартную кодировку. Попробуйте загрузить другой файл.")
        #     print("!!!ВОЗНИКЛА ОШИБКА ПРИ ИНДЕКСАЦИИ!!!\nОШИБКА:", e)
        #     folder_index_local = None

    if folder_index_local is None:
        st.stop()
    st.session_state.files = [file.id for file in files_var]
    st.session_state.folder_index = folder_index_local
    return folder_index_local


def chunk_files_func(files_var, chunk_size, chunk_overlap):
    chunked_files_local = []
    chunked_docs_local = []
    for file in tqdm(files_var, desc="chunking files"):
        chunked_file, chunked_doc = chunk_file(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_files_local.append(chunked_file)
        chunked_docs_local += (chunked_doc)
    return chunked_files_local, chunked_docs_local

def read_files_func(uploaded_files_var):
    files_local = []
    for file in uploaded_files_var:
        try:
            files_local.append(read_file(file))
        except Exception as e:
            display_file_read_error(e, file_name=file.name)
    return files_local

def upload_and_settings():
    uploaded_files = st.file_uploader(
        "Загрузите ваши документы",
        type=["pdf", "docx", "txt", "xlsx", "pptx", "doc", "xlsm", "xls"],
        help="Сканированные документы пока не поддерживаются.",
        accept_multiple_files=True,
    )

    if len(uploaded_files) > MAX_LINES:
        st.warning(f"Достигнуто максимальное количество файлов. Только первые {MAX_LINES} будут обработаны.")
        uploaded_files = uploaded_files[:MAX_LINES]

    with st.expander("Расширенные настройки"):
        return_all_chunks = st.checkbox("Показывать все источники",
                                        help="Показывать все фрагменты из ваших документов, которые могли быть использованы для генерации ответа. (не работает в режиме ChatBot)")

        option = st.selectbox(
            "Выберите тип поиска",
            ["Поиск смысловой информации, выявление ключевых концепций",
             "Поиск конкретной информации, анализ фактов/данных"],
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
            chunk_size_input = 3000
            chunk_overlap_input = 400
            num_chunks = 5
        elif option == "Поиск конкретной информации, анализ фактов/данных":
            chunk_size_input = 100
            chunk_overlap_input = 100
            num_chunks = 10
        else:
            chunk_size_input = 1000
            chunk_overlap_input = 400
            num_chunks = 5

        show_full_doc = False

    return uploaded_files, return_all_chunks, option, model, chunk_size_input, chunk_overlap_input, num_chunks, show_full_doc

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

st.set_page_config(page_title="RAG_MVP", page_icon="🔍", layout="wide")
# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
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

metrika = """
<!-- Yandex.Metrika counter -->
<script type="text/javascript" >
   (function(m,e,t,r,i,k,a){m[i]=m[i]function(){(m[i].a=m[i].a[]).push(arguments)};
   m[i].l=1*new Date();
   for (var j = 0; j < document.scripts.length; j++) {if (document.scripts[j].src === r) { return; }}
   k=e.createElement(t),a=e.getElementsByTagName(t)[0],k.async=1,k.src=r,a.parentNode.insertBefore(k,a)})
   (window, document, "script", "https://mc.yandex.ru/metrika/tag.js", "ym");

   ym(99478767, "init", {
        clickmap:true,
        trackLinks:true,
        accurateTrackBounce:true,
        webvisor:true
   });
</script>
<noscript><div><img src="https://mc.yandex.ru/watch/99478767" style="position:absolute; left:-9999px;" alt="" /></div></noscript>
<!-- /Yandex.Metrika counter -->
"""
st.markdown(hide_default_format, unsafe_allow_html=True)
st.markdown(metrika, unsafe_allow_html=True)


# Add a toggle switch for chatbot mode
sidebar()

chatbot_mode = st.sidebar.checkbox("ChatBot Mode (beta)", help="Включить режим чат-бота для диалога с пользователем.")
float_init()

# /////////////////////////////////////////////////////////////////////////////////////
# THIS IS AIMATE CHATBOT MAIN PAGE HERE:
if chatbot_mode:
    st.header("AimateChatBot")

    uploaded_files_chat, return_all_chunks_chat, option_chat, model_chat, chunk_size_input_chat, chunk_overlap_input_chat, num_chunks_chat, show_full_doc_chat = upload_and_settings()

    if not any(uploaded_file for uploaded_file in uploaded_files_chat):
        st.stop()

    files_chat = read_files_func(uploaded_files_chat)

    if sum(len(doc.page_content) for file in files_chat for doc in file.docs) > 100_000_000_000:
        st.warning("Ваши файлы содержат слишком много текста. Пожалуйста, загрузите файлы поменьше.")
        st.stop()

    chunked_files_chat, chunked_docs_chat = chunk_files_func(files_chat, chunk_size=chunk_size_input_chat, chunk_overlap=chunk_overlap_input_chat)

    if not any(is_file_valid(chunked_file) for chunked_file in chunked_files_chat):
        st.stop()

    if 'files' not in st.session_state:
        st.session_state.files = []

    if 'folder_index' not in st.session_state:
        st.session_state.folder_index = None

    if [file.id for file in chunked_files_chat] != st.session_state.files:
        print("Creating new folder index")
        folder_index_chat = create_folder_index(chunked_files_chat, EMBEDDING, VECTOR_STORE)
        st.success("Документы успешно обработаны!")

    else:
        print("Using existing folder index")
        folder_index_chat = st.session_state.folder_index
        st.success("Документы успешно обработаны!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(str(message["content"]))

    # add a button to clear chat history

    if st.session_state.messages:
        button_container = st.container()
        button_container.float("bottom: 140px; right: -825px;")
        with button_container:
            if st.button("Очистить историю", type="primary"):
                st.session_state.messages = []
                st.rerun()

    if prompt:=st.chat_input("Задайте свой вопрос по документам"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        llm = get_llm(model=model_chat)

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ на ваш вопрос в документации..."):
                history = st.session_state.messages
                if len(history) > 0:
                    line = ""
                    for message in history:
                        line += str(message["role"]) + ": " + str(message["content"]) + "\n"
                    history = line
                else:
                    history = ""
                
                result = query_folder(
                    folder_index=folder_index_chat,
                    chunked_files=chunked_docs_chat,
                    query=prompt,
                    history=history,
                    return_all=return_all_chunks_chat,
                    llm=llm,
                    num_sources=num_chunks_chat,
                )

                response = st.write_stream(result)

        st.session_state.messages.append({"role": "assistant", "content": response})


# /////////////////////////////////////////////////////////////////////////////////////
# THIS IS AIMATE DOCS MAIN PAGE HERE:
else:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(cur_dir, "aimate_symbol_cropped.png")
    st.image(logo_path, width=100)
    st.header("Умный поиск по документации AimateDocs")

    main_col1, main_col2 = st.columns([0.45, 0.55], gap='medium')

    with main_col2:
        with st.form(key="qa_form", border=False):
            query = st.text_area("Задайте вопрос по загруженным документам", height=94)
            st.session_state.query = query
            submit = st.form_submit_button("Отправить")

    with main_col1:
        uploaded_files, return_all_chunks, option, model, chunk_size_input, chunk_overlap_input, num_chunks, show_full_doc = upload_and_settings()

        if not any(uploaded_file for uploaded_file in uploaded_files):
            if submit:
                st.error("Прежде чем отправить запрос, загрузите документы.")
            st.stop()

        files = read_files_func(uploaded_files)

        if sum(len(doc.page_content) for file in files for doc in file.docs) > 100_000_000_000:
            st.warning("Ваши файлы содержат слишком много текста. Пожалуйста, загрузите файлы поменьше.")
            st.stop()

        chunked_files, chunked_docs = chunk_files_func(files, chunk_size=chunk_size_input, chunk_overlap=chunk_overlap_input)

        if not any(is_file_valid(chunked_file) for chunked_file in chunked_files):
            st.stop()

        if 'files' not in st.session_state:
            st.session_state.files = []

        if 'folder_index' not in st.session_state:
            st.session_state.folder_index = None

        if [file.id for file in chunked_files] != st.session_state.files:
            print("Creating new folder index")
            folder_index = create_folder_index(chunked_files, EMBEDDING, VECTOR_STORE)
            st.success("Документы успешно обработаны!")
        else:
            print("Using existing folder index")
            folder_index = st.session_state.folder_index
            st.success("Документы успешно обработаны!")

    with main_col2:
        if submit:
            if not is_query_valid(query):
                st.stop()

            answer_col, sources_col = st.columns(2)

            llm = get_llm(model=model)

            with st.spinner("Ищем ответ на ваш вопрос в документации..."):
                result = query_folder(
                    folder_index=folder_index,
                    query=query,
                    chunked_files=chunked_docs,
                    history="",
                    return_all=return_all_chunks,
                    llm=llm,
                    num_sources=num_chunks,
                )

            with answer_col:
                st.markdown("#### Ответ")
                response = st.write_stream(result)

            with sources_col:
                st.markdown("#### Источники")
                for source in result.sources:
                    st.markdown(source.page_content)
                    st.markdown(source.metadata["source"])
                    st.markdown("---")