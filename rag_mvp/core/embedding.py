from langchain.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
import time
import logging
import sys
import os
from datetime import datetime

from .parsing import File
from langchain_community.vectorstores import FAISS
from .embedder import MultilingualE5 #Pinecone_MultilingualE5
from langchain.embeddings.base import Embeddings
from langchain_mistralai import MistralAIEmbeddings
from typing import List, Type
from langchain.docstore.document import Document
import streamlit as st
from dataclasses import dataclass

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger(__name__)

@dataclass(init=False)
class FolderIndex:
    """Index for a collection of files (a folder)"""

    def __init__(self, files: List[File], index: VectorStore):
        self.name: str = "default"
        self.files = files
        self.index: VectorStore = index

    @staticmethod
    def _combine_files(files: List[File]) -> List[Document]:
        """Combines all the documents in a list of files into a single list."""

        all_texts = []
        for file in files:
            for doc in file.docs:
                doc.metadata["file_name"] = file.name
                doc.metadata["file_id"] = file.id
                all_texts.append(doc)

        return all_texts

    @classmethod
    def from_files(
        cls, files: List["File"], embeddings, vector_store: Type["VectorStore"]
    ) -> "FolderIndex":
        """Creates or loads a FAISS vector index from files."""
        print("------Combine Files------")
        all_docs = cls._combine_files(files)

        try:
            print("------Creating / Loading Vectorstore------")
            index_start_time = datetime.now()

            if os.path.exists("faiss_index"):
                # Загружаем уже существующий индекс
                index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                
                # # Добавляем новые документы к индексу
                # new_index = vector_store.from_documents(
                #     documents=all_docs,
                #     embedding=embeddings,
                # )
                # index.merge_from(new_index)

                # # Сохраняем обновлённый индекс
                # index.save_local("faiss_index")
            else:
                # Если индекса нет — создаём новый и сохраняем
                index = vector_store.from_documents(
                    documents=all_docs,
                    embedding=embeddings,
                )
                index.save_local("faiss_index")

            index_end_time = datetime.now() - index_start_time
            print(f"------Vectorstore Ready: {index_end_time.seconds}s------")

        except KeyError as e:
            print(f"Error: {e}")
            raise e

        return cls(files=files, index=index)  
    

@st.cache_resource(show_spinner=False, ttl="8h")
def get_model(embedding: str, **kwargs):
    if embedding == 'mistral':
        return MultilingualE5()
        # return MistralAIEmbeddings(**kwargs)
    if embedding == 'multilinguale5':
        return MultilingualE5()


def embed_files(
    files: List[File], embedding: str, vector_store: str, **kwargs
) -> FolderIndex:
    """Embeds a collection of files and stores them in a FolderIndex."""


    supported_vector_stores: dict[str, Type[VectorStore]] = {
        "faiss": FAISS,
    }

    _embeddings = get_model(embedding, **kwargs)

    if vector_store in supported_vector_stores:
        _vector_store = supported_vector_stores[vector_store]
    else:
        raise NotImplementedError(f"Vector store {vector_store} not supported.")

    return FolderIndex.from_files(
        files=files, embeddings=_embeddings, vector_store=_vector_store
    )
