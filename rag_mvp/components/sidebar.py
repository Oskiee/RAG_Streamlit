import streamlit as st

from dotenv import load_dotenv
from .faq import faq
import os

load_dotenv()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown(
            "## :orange_book: Использование \n"
            "1. Загрузите ваши документы по которым будет производиться поиск :ledger:\n"
            "2. Задайте интересующий вас вопрос по загруженным документам :question:\n"
            "3. Вы также можете изменять параметры расширенных настроек 🛠️\n\n"
        )

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        style_path = os.path.join(curr_dir, "style.css")
        local_css(style_path)

        with st.expander("**FAQ**"):
            faq()


