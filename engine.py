import json
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from config import (
    llm, engine, index, FilterOperator, MetadataFilter, 
    MetadataFilters, DB_URL
)
from llama_index.core.prompts import PromptTemplate

# --- Shared DB Tools ---

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
    
    import psycopg2
    import sys
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        ids = [int(r[0]) for r in rows]
        cur.close()
        conn.close()
        print(f"[DEBUG] Narrowing result: {len(ids)} candidate IDs found.\n")
        sys.stdout.flush()
        return ids
    except Exception as e:
        print(f"[DEBUG] SQL Error in sql_candidate_ids: {str(e)}")
        sys.stdout.flush()
        return []

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

# --- State Definition ---
class DiscoveryState(TypedDict):
    query: str
    intent: Dict[str, Any]
    candidate_ids: List[int]
    refined_audience: List[Dict[str, Any]]
    validation: Dict[str, Any]
    enriched_profiles: List[Dict[str, Any]]
    recommendation: str
    iterations: int
    error: str

# --- Nodes ---

async def intent_classification_node(state: DiscoveryState):
    """Step 1: Parse the user query into a structured intent."""
    print("[NODE] Intent Classification")
    
    prompt = PromptTemplate(
        "You are an expert audience strategist. Parse the following request into a structured JSON intent.\n"
        "User Request: {query}\n\n"
        "Output JSON with these keys:\n"
        "- theme: The campaign focus (e.g. 'luxury red fashion')\n"
        "- product_types: List of products mentioned (e.g. ['socks', 'jacket'])\n"
        "- colors: List of colors mentioned\n"
        "- time_window_days: Number of days for activity (default 30)\n"
        "- min_spend: Minimum total spend if implied (default 0)\n"
        "- focus_keywords: Key semantic descriptors for vector search\n"
        "- feedback: Instructions for refinement (initially empty)\n\n"
        "JSON output only:"
    )
    
    response = await llm.acomplete(prompt.format(query=state['query']))
    # Basic JSON extraction
    text = response.text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        intent = json.loads(text)
    except:
        intent = {"theme": state['query'], "product_types": [], "time_window_days": 30}
    
    return {"intent": intent}

async def behavioral_sql_gate_node(state: DiscoveryState):
    """Step 2: Narrow candidates via SQL using LLM."""
    print("[NODE] Behavioral SQL Gate")
    intent = state['intent']
    iterations = state.get('iterations', 0)
    
    # Update iterations count in state
    new_iterations = iterations + 1
    
    # Adaptive time window
    days = intent.get('time_window_days', 30)
    if iterations > 0:
        days = days * (iterations + 1)
        print(f"[DEBUG] Widening search window to {days} days (Iteration {iterations})")

    prompt = PromptTemplate(
        "You are a SQL expert. Generate a valid PostgreSQL WHERE clause snippet for the 'events' table based on this user intent.\n"
        "TABLE SCHEMA: 'events' table with columns: customer_id (bigint), event_type (text: 'view', 'purchase'), product (text), color (text), event_timestamp (timestamp).\n\n"
        "INTENT:\n"
        "- product_types: {product_types}\n"
        "- colors: {colors}\n"
        "- time_window_days: {days}\n\n"
        "RULES:\n"
        "1. Output ONLY the WHERE clause snippet (do not include the 'WHERE' keyword itself).\n"
        "2. Use 'ILIKE' with wildcards for product matching to handle plural/singular (e.g., product ILIKE 'sock%').\n"
        "3. Cast event_timestamp explicitly if comparing (e.g., event_timestamp::timestamp >= now() - interval '{days} days').\n"
        "4. Combine conditions with 'AND'.\n\n"
        "SQL Snippet:"
    )
    
    formatted_prompt = prompt.format(
        product_types=intent.get('product_types', []),
        colors=intent.get('colors', []),
        days=days
    )
    
    response = await llm.acomplete(formatted_prompt)
    where_clause = response.text.strip()
    
    # Strip markdown code blocks if present
    if "```sql" in where_clause:
        where_clause = where_clause.split("```sql")[1].split("```")[0].strip()
    elif "```" in where_clause:
        where_clause = where_clause.split("```")[1].split("```")[0].strip()
    
    # Surgical cleanup: remove leading/trailing noise but KEEP internal quotes
    where_clause = where_clause.replace("WHERE ", "").strip().strip(";").strip('"')
    
    # Failsafe: Convert double quotes to single quotes for SQL values
    if '"' in where_clause:
        print(f"[DEBUG] Sanitizing SQL WHERE clause (double to single quotes)...")
        where_clause = where_clause.replace('"', "'")
        
    print(f"[DEBUG] Final SQL Gate WHERE clause: {where_clause}")
    
    try:
        ids = sql_candidate_ids(where_clause)
        return {"candidate_ids": ids, "iterations": new_iterations}
    except Exception as e:
        return {"error": f"SQL Gate Error: {str(e)}", "candidate_ids": [], "iterations": new_iterations}

async def semantic_refinement_node(state: DiscoveryState):
    """Step 3: Rank results via Vector Search."""
    print("[NODE] Semantic Refinement")
    if not state.get('candidate_ids'):
        return {"refined_audience": []}
        
    theme = state['intent'].get('theme', state['query'])
    
    # Metadata filters using the candidate IDs from SQL gate
    # Based on curl, customer_id is at top level of payload
    metadata_filters = [
        MetadataFilter(key="customer_id", value=state['candidate_ids'], operator=FilterOperator.IN)
    ]
    
    # Optional luxury filter if theme implies it
    if "luxury" in theme.lower():
        metadata_filters.append(
            MetadataFilter(key="metadata.likes_luxury", value=1, operator=FilterOperator.EQ)
        )
        
    filters_obj = MetadataFilters(filters=metadata_filters)
    
    try:
        query_engine = index.as_query_engine(similarity_top_k=100, filters=filters_obj)
        response = await query_engine.aquery(theme)
        
        # For a robust graph, let's just use the 'raw' customer records from the response source nodes
        refined = []
        for node in response.source_nodes:
            # LlamaIndex stores the original payload in node.metadata
            payload = node.metadata
            if 'metadata' in payload:
                customer_data = payload['metadata']
            else:
                customer_data = payload
                
            refined.append({
                "customer_id": customer_data.get('customer_id'),
                "score": node.score,
                "text": node.get_content(),
                "data": customer_data
            })
            
        return {"refined_audience": refined}
    except Exception as e:
        print(f"[DEBUG] Error in semantic refinement: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"refined_audience": [], "error": f"Semantic Error: {str(e)}"}

async def audience_validation_node(state: DiscoveryState):
    """Step 4: Check if audience meets campaign requirements."""
    print("[NODE] Audience Validation")
    audience = state['refined_audience']
    count = len(audience)
    
    validation = {"status": "ok", "message": ""}
    
    if count < 5 and state['iterations'] < 3:
        validation = {"status": "too_small", "message": "Widen criteria. Increase time window or remove specific color filters."}
    elif count > 100:
        validation = {"status": "too_large", "message": "Tighten criteria. Increase min_spend or add more specific product filters."}
    
    if validation["status"] != "ok":
        print(f"[DEBUG] Validation failed: {validation['message']} (Iteration {state['iterations']}/3)")
        
    return {"validation": validation}

def should_continue(state: DiscoveryState):
    """Router for validation loop."""
    if state['validation']['status'] == "ok":
        return "enrich"
    return "sql_gate"

async def profile_enrichment_node(state: DiscoveryState):
    """Step 5: Fetch full CRM profiles for the top results."""
    print("[NODE] Profile Enrichment")
    # Handle the refined structure with 'customer_id'
    ids = []
    for p in state['refined_audience'][:20]:
        cid = p.get('customer_id')
        if cid:
            ids.append(int(cid))
            
    if not ids:
        return {"enriched_profiles": []}
        
    query = f"SELECT * FROM customers WHERE customer_id IN ({', '.join(map(str, ids))})"
    profiles = raw_data_query(query)
    return {"enriched_profiles": profiles}

async def campaign_recommendation_node(state: DiscoveryState):
    """Step 6: Generate campaign strategy."""
    print("[NODE] Campaign Recommendation")
    
    prompt = PromptTemplate(
        "You are a CMO. Based on this audience segment, generate a precision campaign strategy.\n"
        "Audience Summary: {audience_summary}\n"
        "Intent: {intent}\n\n"
        "Provide:\n"
        "1. Target Audience Descriptor\n"
        "2. Messaging Strategy\n"
        "3. Recommended Channels\n"
        "4. Expected Impact\n"
    )
    
    summary = f"{len(state['enriched_profiles'])} high-value customers matching {state['intent']['theme']}"
    response = await llm.acomplete(prompt.format(audience_summary=summary, intent=state['intent']))
    
    return {"recommendation": response.text}

async def output_node(state: DiscoveryState):
    """Step 7: Persist results to disk."""
    print("[NODE] Output Node")
    
    with open("audience.json", "w") as f:
        json.dump(state['enriched_profiles'], f, indent=2)
        
    with open("campaign.md", "w") as f:
        f.write("# Campaign Strategy\n\n")
        f.write(state['recommendation'])
        
    return {"status": "completed"}

# --- Graph Construction ---

workflow = StateGraph(DiscoveryState)

workflow.add_node("intent", intent_classification_node)
workflow.add_node("sql_gate", behavioral_sql_gate_node)
workflow.add_node("semantic", semantic_refinement_node)
workflow.add_node("validate", audience_validation_node)
workflow.add_node("enrich", profile_enrichment_node)
workflow.add_node("recommend", campaign_recommendation_node)
workflow.add_node("output", output_node)

workflow.set_entry_point("intent")
workflow.add_edge("intent", "sql_gate")
workflow.add_edge("sql_gate", "semantic")
workflow.add_edge("semantic", "validate")

workflow.add_conditional_edges(
    "validate",
    should_continue,
    {
        "enrich": "enrich",
        "sql_gate": "sql_gate"
    }
)

workflow.add_edge("enrich", "recommend")
workflow.add_edge("recommend", "output")
workflow.add_edge("output", END)

app = workflow.compile()

async def run_discovery_pipeline(query: str):
    inputs = {"query": query, "iterations": 0}
    final_state = await app.ainvoke(inputs)
    return final_state
