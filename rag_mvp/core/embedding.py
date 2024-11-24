from langchain.vectorstores import VectorStore
import time

from .parsing import File
from langchain_community.vectorstores import FAISS
from .embedder import MultilingualE5
from langchain.embeddings.base import Embeddings
from langchain_mistralai import MistralAIEmbeddings
from typing import List, Type
from langchain.docstore.document import Document
import streamlit as st
from dataclasses import dataclass


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
        cls, files: List[File], embeddings: Embeddings, vector_store: Type[VectorStore]
    ) -> "FolderIndex":
        """Creates an index from files."""
        all_docs = cls._combine_files(files)
        try:
            index = vector_store.from_documents(
                    documents=all_docs,
                    embedding=embeddings,
                )
        except KeyError as e:
            print(f"Error: {e}")
            raise e
        # for file in files[1:]:
        #     all_docs_temp = cls._combine_files([file])
        #
        #     try:
        #         index_temp = vector_store.from_documents(
        #             documents=all_docs_temp,
        #             embedding=embeddings,
        #         )
        #     except KeyError as e:
        #         print(f"Error: {e}")
        #         raise e
        #
        #     index.merge_from(index_temp)
        #
        #     time.sleep(5)

        return cls(files=files, index=index)

@st.cache_resource(show_spinner=False, ttl="8h")
def get_model(embedding: str, **kwargs):
    if embedding == 'mistral':
        return MistralAIEmbeddings(**kwargs)
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
