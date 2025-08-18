from sentence_transformers import SentenceTransformer
from typing import List
from langchain_core.embeddings import Embeddings
from core import config
import torch
import numpy as np
from pinecone import Pinecone, PineconeApiException
import logging
import time

logger = logging.getLogger(__name__)


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class MultilingualE5(Embeddings):
    def __init__(self, model_name="multilingual-e5-large"):
        self.model = Pinecone(api_key=config.PINECONE_API_KEY, )
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        num_seq = len(texts)
        batches = int(np.ceil(num_seq / 96))
        
        for i in range(batches):
            if i == batches - 1:
                batch = texts[(i*96):num_seq]
            else:
                batch = texts[(i*96):(i*96+96)]
            try:
                data = self.model.inference.embed(
                    model="multilingual-e5-large",
                    inputs=batch,
                    parameters={"input_type": "passage", },
                    ).data
            except PineconeApiException as e:
                logger.error(f"Error while making embedding of a batch. Retrying in a minute.")
                time.sleep(60)
                data = self.model.inference.embed(
                    model="multilingual-e5-large",
                    inputs=batch,
                    parameters={"input_type": "passage", },
                    ).data
            
            result += [d['values'] for d in data]

            time.sleep(1)  # Sleep to avoid rate limiting
        
        return result

    def embed_query(self, query: str) -> List[float]:
        queries = [query]
        embedding = self.model.inference.embed(
            model='multilingual-e5-large',
            inputs=queries,
            parameters={
                    "input_type": "query"
                }
            )
        return embedding.data[0]['values']
    # def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
    #     self.model = SentenceTransformer(model_name, trust_remote_code=True)
    #     self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    # def embed_documents(self, texts: List[str]) -> List[List[float]]:
    #     return [self.model.encode(get_detailed_instruct(self.task, text), normalize_embeddings=True).tolist() for text in texts]

    # def embed_query(self, query: str) -> List[float]:
    #     encoded_query = self.model.encode(query)
    #     return encoded_query.tolist()
    

# class Pinecone_MultilingualE5(Embeddings):
#     def __init__(self, model_name="multilingual-e5-large"):
#         self.model = Pinecone(api_key=config.PINECONE_API_KEY, )
#         self.task = 'Given a web search query, retrieve relevant passages that answer the query'

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         data = self.model.inference.embed(
#             model="multilingual-e5-large",
#             inputs=get_detailed_instruct(texts),
#             parameters={"input_type": "passage"},
#             ).data
        
#         return [d['values'] for d in data]

#     def embed_query(self, query: str) -> List[float]:
#         queries = [query]
#         embedding = self.model.inference.embed(
#             model='multilingual-e5-large',
#             inputs=queries,
#             parameters={
#                     "input_type": "query"
#                 }
#             )
#         return embedding.data[0]['values'].tolist()


class HybridSearch:
    def __init__(self, pinecone_api_key = config.PINECONE_API_KEY, model_name=config.EMBEDDING_MODEL_NAME):
        self.pc = pinecone_api_key

    def process_text(self, text):
        """Преобразует входной текст в векторное представление (эмбеддинг) с помощью модели.

        Функция принимает текстовую строку, отправляет ее в модель для получения векторного
        представления и возвращает результат в виде тензора PyTorch.

        Args:
            text (str): Входной текст для обработки. Должен быть непустой строкой.

        Returns:
            torch.Tensor: Тензор с векторным представлением текста. Размерность зависит от модели.

        Examples:
            >>> processor.process_text("Пример текста для обработки")
            tensor([0.1234, -0.5678, ..., 0.9012])
        """
        try:
            queries = [text]
            embedding = self.pc.inference.embed(
                model=self.model_name,
                inputs=queries,
                parameters={
                    "input_type": "query"
                }
            )
            return torch.Tensor(embedding.data[0]["values"])
        except Exception as e:
            logger.error(f"Error while making embedding of a text.\n Text: {text}\n Error: {e}")
            raise




