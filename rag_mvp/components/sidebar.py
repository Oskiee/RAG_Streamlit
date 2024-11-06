import streamlit as st

from dotenv import load_dotenv
from .faq import faq
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
            "## Использование:\n"
            "1. Загрузите pdf, docx, или txt файл📄\n"
            "2. Задайте вопрос по документу💬\n"
            "3. Вы также можете изменять параметры расширенных настроек.\n\n"
            ":orange[Состояние работы отображается в правом верхнем углу экрана. Если вы видите надпись 'RUNNING', это значит, что процесс индексации или генерации ответа еще не завершен, пожалуйста подождите.]\n"
        )
        faq()
