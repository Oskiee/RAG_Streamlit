import streamlit as st

from rag_mvp.components.faq import faq
from dotenv import load_dotenv
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
            "## Использование:\n"
            "1. Загрузите pdf, docx, или txt файл📄\n"
            "2. Задайте вопрос по документу💬\n\n"
            "Вы также можете изменять параметры расширенных настроек.\n\n"
            "Состояние работы отображается в правом верхнем углу экрана. Если вы видите надпись 'RUNNING', это значит, что процесс индексации или генерации ответа еще не завершен, пожалуйста подождите.\n"
        )