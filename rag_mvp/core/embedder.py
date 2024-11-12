import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.embeddings.base import Embeddings

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class MultilingualE5(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(get_detailed_instruct(self.task, text), normalize_embeddings=True).tolist() for text in texts]

    def embed_query(self, query: str) -> List[float]:
        encoded_query = self.model.encode(query)
        return encoded_query.tolist()

# Streamlit cache_resource для кэширования модели
@st.cache_resource(show_spinner=False)
def get_model(model_name="intfloat/multilingual-e5-large-instruct"):
    return MultilingualE5(model_name)