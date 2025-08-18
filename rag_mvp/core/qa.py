from typing import List
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from .prompts import STUFF_PROMPT
from langchain.docstore.document import Document
from .embedding import FolderIndex
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from openai import OpenAI
from core import config
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker


class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]


def query_folder(
    query: str,
    history: str,
    folder_index: FolderIndex,
    chunked_files: list,
    llm: ChatOpenAI,
    return_all: bool = False,
    num_sources: int = 5,):
# ) -> AnswerWithSources:
    """Queries a folder index for an answer.

    Args:
        query (str): The query to search for.
        folder_index (FolderIndex): The folder index to search.
        return_all (bool): Whether to return all the documents from the embedding or
        just the sources for the answer.
        model (str): The model to use for the answer generation.
        **model_kwargs (Any): Keyword arguments for the model.

    Returns:
        AnswerWithSources: The answer and the source documents.
    """

    vector_retriever = folder_index.index.as_retriever(search_kwargs={"k": config.HYBRID_SEARCH_TOP_K})
    bm25_retriever = BM25Retriever.from_documents(chunked_files)
    bm25_retriever.k = config.HYBRID_SEARCH_TOP_K
    # reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[config.HYBRID_SEARCH_KEYWORD_WEIGHT, config.HYBRID_SEARCH_VECTOR_WEIGHT]
        )
    
    # compressor = CrossEncoderReranker(model=reranker, top_n=5)

    # compression_retriever = ContextualCompressionRetriever(
    #     base_retriever=ensemble_retriever,
    #     base_compressor=ensemble_retriever
    # )

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=STUFF_PROMPT)
    chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)

    # chain = create_retrieval_chain(
    #     llm=llm,
    #     chain_type="stuff",
    #     prompt=STUFF_PROMPT,
    # )


    summary = [
        (
            "system",
            f"""Given a context of recent chat history, summarize the user's question as a search term. Return ONLY this paraphrase.
            Also you should enhance user's question with the chat history if it is necessary and relevant.
            IF YOU THINK USER QUERY IS NOT A SEARCH TERM BUT A QUESTION TO THE CHAT BOT YOU SHOULD LEAVE IT UNCHANGED!
            ALSO, KEEP IN MIND THAT QUERIES CAN REFER TO DIFFERENT DOCUMENTS, SO NOT EVERY QUERY IS DEPENDENT ON PREVIOUS QUESTIONS!
            
            HISTORY: {history}
            QUESTION: {query}"""

        )
    ]

    search_query = llm.invoke(summary).content

    # print(search_query)

    # relevant_docs = folder_index.index.similarity_search(search_query, k=num_sources)
    # relevant_docs = ensemble_retriever.
    # result = chain.invoke(
    #     {"input_documents": relevant_docs, "question": query, "history": history}, return_only_outputs=True
    # )
    # result = chain.invoke({"input": search_query, "question": query, "history": history}, 
    #                       return_only_outputs=True)
    # sources = relevant_docs
    # sources=[]
    # result = chain.invoke({"question": query, "context": history, "summaries": search_query})

    # print(result)


    # if not return_all:
    #     sources = get_sources(result["answer"], folder_index)

    

    for res in chain.stream({"input": search_query, "question": query, "history": history},): 
        if "answer" in res.keys():
            yield res["answer"]

    retrieved_docs = ensemble_retriever.get_relevant_documents(search_query)

    yield f"\n\nИсточники: "

    for i, doc in enumerate(retrieved_docs):
        if i < 5:
            yield f"\n{doc.metadata.get('source')}, "
    # answer = result["answer"].split("SOURCES: ")[0]

    # return {"answer": answer}


def get_sources(answer: str, folder_index: FolderIndex) -> List[Document]:
    """Retrieves the docs that were used to answer the question the generated answer."""

    source_keys = [s for s in answer.split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for file in folder_index.files:
        for doc in file.docs:
            if doc.metadata["source"] in source_keys:
                source_docs.append(doc)
    return source_docs

