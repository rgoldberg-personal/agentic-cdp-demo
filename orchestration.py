import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

load_dotenv()

# --- Configuration ---
DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "synthetic_documents")

# --- Cost Observability ---
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode # Fallback tokenizer
)
callback_manager = CallbackManager([token_counter])

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

# --- LlamaIndex Components ---
engine = create_engine(DB_URL)
sql_database = SQLDatabase(engine, include_tables=["customers", "events"])
sql_query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=["customers", "events"])

from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

# 2. Vector Layer
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
aclient = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vector_store = QdrantVectorStore(
    client=client, 
    aclient=aclient, 
    collection_name=COLLECTION_NAME
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 3. Hybrid/Raw Tools
def sql_candidate_ids(where_clause: str):
    """
    SQL NARROWING GATE:
    Filters the population based on behavioral event criteria.
    - Input: A raw SQL WHERE clause for the 'events' table.
    - Logic: Executes 'SELECT DISTINCT customer_id FROM events WHERE {where_clause}'.
    - Use case: Narrowing discovery to only people who bought a specific product, 
      viewed a specific color, or transacted in a time window.
    """
    query = f"SELECT DISTINCT customer_id FROM events WHERE {where_clause}"
    print(f"\n[DEBUG] SQL narrowing tool starting...")
    print(f"[DEBUG] Query: {query}")
    
    import pandas as pd
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        ids = df["customer_id"].tolist()
        print(f"[DEBUG] Narrowing result: {len(ids)} candidate IDs found.\n")
        return ids

def raw_data_query(query: str):
    """
    RAW DATA RETRIEVER:
    Retrieves complete, campaign-ready customer profiles.
    - Input: A valid PostgreSQL SELECT query for the 'customers' table.
    - Use case: Returning structured JSON data, email lists, or full details 
      after an audience has been identified using analytics or discovery tools.
    """
    print(f"\n[DEBUG] SQL data retriever starting...")
    print(f"[DEBUG] Query: {query}")
    
    with engine.connect() as conn:
        import pandas as pd
        df = pd.read_sql(query, conn)
        result = df.to_dict(orient='records')
        print(f"[DEBUG] Data retrieval result: {len(result)} records found.\n")
        return result

async def hybrid_audience_search(query: str, sql_where: str = None, filters_dict: dict = None):
    """
    TRUE HYBRID AUDIENCE DISCOVERY:
    The architectural core for combining behavior (SQL) with intent (Vector).
    - query: Semantic intent (e.g., 'interested in high-end lifestyle').
    - sql_where: Behavioral SQL filter for the events table (e.g., 'product=\"socks\"').
    - filters_dict: Structured CRM metadata constraints (e.g., {"country": "PL"}).
    
    LOGIC:
    1. SQL narrow: If 'sql_where' is provided, it narrows candidate_ids using sql_candidate_ids.
    2. Vector refine: Performs semantic search strictly inside the candidate IDs 
       and applies structured CRM metadata filters.
    """
    print(f"\n[DEBUG] Hybrid Audience Discovery starting...")
    print(f"[DEBUG] Semantic Query: {query}")
    if sql_where: print(f"[DEBUG] SQL Narrowing Clause: {sql_where}")
    if filters_dict: print(f"[DEBUG] Metadata Filters: {filters_dict}")

    candidate_ids = None

    # Step 1 — SQL narrowing (Behavioral Gate)
    if sql_where:
        try:
            candidate_ids = sql_candidate_ids(sql_where)
            if not candidate_ids:
                return "No customers match the specified behavioral SQL conditions."
        except Exception as e:
            return f"SQL Error in narrowing: {str(e)}"

    # Step 2 — Vector filtering (Semantic Refinement)
    metadata_filters = []
    
    # Apply SQL candidates if any
    if candidate_ids is not None:
        metadata_filters.append(
            MetadataFilter(key="metadata.customer_id", value=candidate_ids, operator=FilterOperator.IN)
        )
    
    # Apply other structured CRM constants if any
    if filters_dict:
        for key, value in filters_dict.items():
            processed_value = value
            if isinstance(value, str):
                if value.isdigit():
                    processed_value = int(value)
                else:
                    try: processed_value = float(value)
                    except ValueError: pass
            metadata_filters.append(
                MetadataFilter(key=f"metadata.{key}", value=processed_value, operator=FilterOperator.EQ)
            )
    
    filters_obj = MetadataFilters(filters=metadata_filters) if metadata_filters else None
    
    print(f"[DEBUG] Vector filtering active: {len(metadata_filters)} filters applied.")
    query_engine = index.as_query_engine(similarity_top_k=10, filters=filters_obj)
    response = await query_engine.aquery(query)
    print(f"[DEBUG] Hybrid search complete.\n")
    return str(response)

from llama_index.core.tools import FunctionTool

# --- Tools ---
tools = [
    QueryEngineTool(
        query_engine=sql_query_engine,
        metadata=ToolMetadata(
            name="sql_analytics",
            description=(
                "Use this tool for analytical questions like 'how many', 'counts', or 'sums'. "
                "Translates natural language directly to SQL for deterministic CRM analysis."
            )
        ),
    ),
    FunctionTool.from_defaults(
        async_fn=hybrid_audience_search,
        name="hybrid_discovery",
        description=(
            "Use this for TRUE hybrid queries involving behavioral event filters (sql_where) "
            "combined with semantic intent (query). This is the primary discovery engine."
        )
    ),
    FunctionTool.from_defaults(
        fn=raw_data_query,
        name="sql_data_retriever",
        description=(
            "Use this tool to fetch detailed JSON customer profiles after segments are identified. "
            "Available columns in customers: customer_id, first_name, last_name, email, country, age, total_spent, favorite_color."
        )
    )
]

# --- Agent ---
SYSTEM_PROMPT = """You are an AI Audience Discovery Expert for a CDP, powered by LlamaIndex.
Your goal is to provide accurate audience segments using your specialized tools.

LLAMAINDEX ARCHITECTURE:
You use LlamaIndex to bridge the gap between structured SQL data and semantic vector embeddings.

TRUE HYBRID DISCOVERY (The priority flow):
When a user asks for an audience based on behavior (purchases, views) and interests:
1. Identify behavioral SQL conditions for the 'sql_where' parameter (e.g., product='socks').
2. Identify semantic intent for the 'query' parameter (e.g., "luxury enthusiasts").
3. Call 'hybrid_discovery' to get the refined segment.
4. If details/JSON are requested, use 'sql_data_retriever' with the resulting customer_ids.

CORE RULES:
- ALWAYS respond in English.
- NEVER assume data. ALWAYS use your tools.
- Use 'hybrid_discovery' for discovery. Use 'sql_analytics' for counting/metrics.
"""

agent = ReActAgent(
    tools=tools, 
    llm=llm, 
    system_prompt=SYSTEM_PROMPT,
    verbose=True
)

async def run_query_async(query: str):
    token_counter.reset_counts()
    response = await agent.run(user_msg=query)
    
    # Report Usage
    print(f"\n[COST OBSERVABILITY]")
    print(f"LLM Prompt Tokens: {token_counter.prompt_llm_token_count}")
    print(f"LLM Completion Tokens: {token_counter.completion_llm_token_count}")
    print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")
    print(f"Total LLM Tokens: {token_counter.total_llm_token_count}\n")
    
    return str(response)

def run_query(query: str):
    return asyncio.run(run_query_async(query))

if __name__ == "__main__":
    async def main():
        print("Testing Advanced Hybrid Fusion...")
        res = await run_query_async("Find customers who bought red socks and are interested in luxury. Return as JSON.")
        print(f"Response: {res}\n")

    asyncio.run(main())
