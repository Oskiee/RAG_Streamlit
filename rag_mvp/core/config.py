# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
# Загружаем .env файл из директории выше 'app'
dotenv_path = APP_DIR.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Переменные окружения ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Модели ---
MISTRAL_MODEL_NAME = "mistral-large-latest"
OPENROUTER_MODEL_NAME= "qwen/qwen-2.5-72b-instruct" #"qwen/qwen-2.5-72b-instruct"
EMBEDDING_MODEL_NAME = "multilingual-e5-large"
RERANKER_MODEL_NAME = 'bge-reranker-v2-m3'
TOKENIZER_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
CHUNKING_MODEL_NAME = "gpt-3.5-turbo"

# --- Проверка наличия необходимых переменных окружения ---
REQUIRED_ENV_VARS = ["OPENROUTER_API_KEY", "PINECONE_API_KEY"] # "TELEGRAM_TOKEN",
missing_vars = [var for var in REQUIRED_ENV_VARS if not globals().get(var)]
if missing_vars:
    # Не будем прерывать выполнение, просто выведем предупреждение
    print(f"Warning: Missing environment variables: {', '.join(missing_vars)}. Some functionality may not work.")
    # raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

print(f"Config loaded.")