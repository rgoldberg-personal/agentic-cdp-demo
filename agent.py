import os
import asyncio
from config import (
    llm, engine, sql_database, index, FilterOperator, MetadataFilter, 
    MetadataFilters, token_counter, Context
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent

# Import discovery engines
from engine import sql_candidate_ids, raw_data_query, run_discovery_pipeline

# --- LlamaIndex Components ---
sql_query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=["customers", "events"])

# --- Tools Logic ---

async def hybrid_audience_search(query: str, sql_where: str = None, filters_dict: dict = None):
    """
    TRUE HYBRID AUDIENCE DISCOVERY:
    The architectural core for combining behavior (SQL) with intent (Vector).
    """
    print(f"\n[DEBUG] Hybrid Audience Discovery starting...")
    print(f"[DEBUG] Semantic Query: {query}")
    if sql_where: 
        if isinstance(sql_where, str) and '"' in sql_where:
            sql_where = sql_where.replace('"', "'")
        print(f"[DEBUG] SQL Narrowing Clause: {sql_where}")
    if filters_dict: print(f"[DEBUG] Metadata Filters: {filters_dict}")

    candidate_ids = None

    if sql_where:
        try:
            candidate_ids = sql_candidate_ids(sql_where)
            if not candidate_ids:
                return "No customers match the specified behavioral SQL conditions."
        except Exception as e:
            return f"SQL Error in narrowing: {str(e)}"

    metadata_filters = []
    
    if candidate_ids is not None:
        metadata_filters.append(
            MetadataFilter(key="customer_id", value=candidate_ids, operator=FilterOperator.IN)
        )
    
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

# --- Tool Definitions ---
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
        async_fn=run_discovery_pipeline,
        name="discovery_expert_pipeline",
        description=(
            "Use this ONLY when the user asks to 'create a campaign', 'develop a strategy', "
            "or 'identify an audience segment' with full reasoning. This is an autonomous "
            "multi-step pipeline that classifications intent, narrows behavior via SQL, "
            "refines via vector search, validates size, and generates a CMO-level strategy."
        )
    ),
    FunctionTool.from_defaults(
        async_fn=hybrid_audience_search,
        name="hybrid_discovery",
        description=(
            "Best for exploratory searches or follow-up questions about specific criteria. "
            "Combines behavioral events with semantic intent."
        )
    ),
    FunctionTool.from_defaults(
        fn=raw_data_query,
        name="sql_data_retriever",
        description=(
            "Use this tool to fetch detailed JSON customer profiles after segments are identified."
        )
    )
]

# --- Agent Configuration ---
SYSTEM_PROMPT = """You are an AI Audience Discovery Expert for a CDP.
Your goal is to provide accurate audience segments by using SQL tools for behavior and Vector tools for intent.

STRATEGY:
- If a user wants to BUILD a campaign, CREATE an audience, or wants a FULL STRATEGY, use 'discovery_expert_pipeline'.
- If a user asks exploratory questions or follow-ups, use 'hybrid_discovery' or 'sql_analytics'.

DATABASE SCHEMA:
- Table 'events': customer_id (bigint), event_type (text: 'view', 'purchase'), product (text), color (text), price (double), event_timestamp (timestamp).
- Table 'customers': customer_id, first_name, last_name, email, country, age, total_spent, favorite_color.

CRITICAL SQL RULES:
- MANDATORY: Use SINGLE QUOTES (') for all string literals.
- NEVER use double quotes (") for values.
- Use 'ILIKE' with '%' for products.
"""

agent = ReActAgent(
    tools=tools, 
    llm=llm, 
    system_prompt=SYSTEM_PROMPT,
    verbose=True
)

agent_context = Context(agent)

def reset_chat():
    global agent_context
    agent_context = Context(agent)
    print("\n[DEBUG] Agent context reset.")

async def chat_async(query: str):
    token_counter.reset_counts()
    handler = agent.run(user_msg=query, ctx=agent_context)
    response = await handler
    return str(response.response.content)

async def run_query_async(query: str):
    global agent_context
    agent_context = Context(agent)
    return await chat_async(query)
