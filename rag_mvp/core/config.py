# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
# Загружаем .env файл из директории выше 'app'
dotenv_path = APP_DIR.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
DB_DIR = APP_DIR / "data"

# --- Переключатели данных ---
USE_HTML_DATA = True
DOCS_CHUNK_SIZE = 2048

# --- Переменные окружения ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ERROR_BOT_TOKEN = os.getenv("TELEGRAM_ERROR_BOT_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Модели ---
MISTRAL_MODEL_NAME = "mistral-large-latest"
OPENROUTER_MODEL_NAME= "qwen/qwen-2.5-72b-instruct" #"qwen/qwen-2.5-72b-instruct"
EMBEDDING_MODEL_NAME = "multilingual-e5-large"
RERANKER_MODEL_NAME = 'bge-reranker-v2-m3'
TOKENIZER_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
CHUNKING_MODEL_NAME = "gpt-3.5-turbo"

# --- Пути к файлам данных ---
# Текстовые версии
TEXT_CHUNK_FILE_PATH = DB_DIR / "chunked_files_pims_text.pkl"
TEXT_EMBEDDING_FILE_PATH = DB_DIR / "doc_embeddings_e5_pims_text.pkl"

# HTML версии
HTML_CHUNK_FILE_PATH = DB_DIR / "chunked_files_pims_html.pkl"
HTML_EMBEDDING_FILE_PATH = DB_DIR / "doc_embeddings_e5_pims_html.pkl"

# --- Активные пути (выбираются переключателем) ---
if USE_HTML_DATA:
    ACTIVE_CHUNK_FILE_PATH = HTML_CHUNK_FILE_PATH
    ACTIVE_EMBEDDING_FILE_PATH = HTML_EMBEDDING_FILE_PATH
    print("Using HTML data files.")
else:
    ACTIVE_CHUNK_FILE_PATH = TEXT_CHUNK_FILE_PATH
    ACTIVE_EMBEDDING_FILE_PATH = TEXT_EMBEDDING_FILE_PATH
    print("Using TEXT data files.")

ACTIVE_DOCUMENTS_FILE_PATH = DB_DIR / f"docs_and_embeddings_e5_pims_documents_{DOCS_CHUNK_SIZE}.pkl"
REFERENCE_TABLE_FILE_PATH = DB_DIR / "refs_table.json"

# Проверка существования активных файлов
if not ACTIVE_CHUNK_FILE_PATH.exists():
    print(f"Warning: Active chunk file not found at {ACTIVE_CHUNK_FILE_PATH}")
if not ACTIVE_EMBEDDING_FILE_PATH.exists():
    print(f"Warning: Active embedding file not found at {ACTIVE_EMBEDDING_FILE_PATH}")


# --- Параметры поиска ---
HYBRID_SEARCH_VECTOR_WEIGHT = 0.6
HYBRID_SEARCH_KEYWORD_WEIGHT = 0.4
HYBRID_SEARCH_TOP_K = 15 # Количество кандидатов до реранкинга
HYBRID_SEARCH_TOP_K_DOCS = 1 # Количество чанков документов для каждого кандидата
RERANKING_TOP_N = 5    # Количество кандидатов после реранкинга (для промпта)
RELEVANCE_CHECK_TOP_K = 3 # Количество источников для показа пользователю
RELEVANCE_CHECK_THRESHOLD = 0.2 # Порог для relevance_check

# --- Проверка наличия необходимых переменных окружения ---
REQUIRED_ENV_VARS = ["MISTRAL_API_KEY", "PINECONE_API_KEY"] # "TELEGRAM_TOKEN",
missing_vars = [var for var in REQUIRED_ENV_VARS if not globals().get(var)]
if missing_vars:
    # Не будем прерывать выполнение, просто выведем предупреждение
    print(f"Warning: Missing environment variables: {', '.join(missing_vars)}. Some functionality may not work.")
    # raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

print(f"Config loaded. DB directory: {DB_DIR}")
print(f"Active chunk file: {ACTIVE_CHUNK_FILE_PATH.name}")
print(f"Active embedding file: {ACTIVE_EMBEDDING_FILE_PATH.name}")