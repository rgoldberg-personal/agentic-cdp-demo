import os
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.workflow import Context
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "synthetic_documents")

# --- Cost Observability ---
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
callback_manager = CallbackManager([token_counter])

# --- LLM & Embedding ---
llm = OpenAILike(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base=os.getenv("OPENROUTER_BASE_URL"),
    is_chat_model=True,
    callback_manager=callback_manager
)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL_NAME"),
    callback_manager=callback_manager
)
Settings.callback_manager = callback_manager

# --- DB Engine & LlamaIndex Components ---
engine = create_engine(DB_URL)
sql_database = SQLDatabase(engine, include_tables=["customers", "events"])

# --- Vector Layer ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
aclient = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vector_store = QdrantVectorStore(
    client=client, 
    aclient=aclient, 
    collection_name=COLLECTION_NAME
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
