```mermaid
flowchart LR

%% =====================
%% User & Agent Layer
%% =====================

U[Marketing / Analyst User]

A[Agentic AI Layer<br/>ReAct Agent]

U --> A

%% =====================
%% Orchestration
%% =====================

A -->|Analytical questions| SQL_TOOL[SQL Analytics Tool]
A -->|Hybrid intent + behavior| HYBRID_TOOL[Hybrid Discovery Tool]
A -->|Profile enrichment| PROFILE_TOOL[SQL Data Retriever]

%% =====================
%% Customer Data Layer
%% =====================

subgraph CDL["Customer Data Layer"]

PG[(PostgreSQL<br/>CRM + Clickstream Events)]
QD[(Qdrant<br/>Customer Embeddings)]

end

SQL_TOOL --> PG
PROFILE_TOOL --> PG

%% =====================
%% True Hybrid Fusion
%% =====================

HYBRID_TOOL -->|"1. Behavioral Gate (SQL)"| PG
PG -->|customer_ids| HYBRID_TOOL
HYBRID_TOOL -->|"2. Semantic Refinement"| QD

QD --> HYBRID_TOOL
HYBRID_TOOL --> A

%% =====================
%% Vector Only Discovery
%% =====================

A -->|Semantic discovery| VECTOR_TOOL[Semantic Search]
VECTOR_TOOL --> QD
QD --> VECTOR_TOOL
VECTOR_TOOL --> A

%% =====================
%% Ingestion Pipeline
%% =====================

subgraph INGEST["Ingestion & Feature Engineering"]

CRM[CRM CSV]
CLICK[Clickstream CSV]

ETL[ETL / Aggregation<br/>Customer Semantic Profiles]

CRM --> ETL
CLICK --> ETL

ETL --> PG
ETL --> QD

end

%% =====================
%% Output
%% =====================

A --> RESULT[Audience Segments<br/>JSON / Explanation]
RESULT --> U
```


# AI Audience Discovery Platform  
**Hybrid Analytical + Semantic CDP Intelligence**

---

## 1. Overview

This project demonstrates an **AI-powered Audience Discovery System** built on top of a Customer Data Platform (CDP) architecture. It bridges the gap between structured relational data and unstructured semantic intent.

**Core Innovation**: A 3-stage "True Hybrid Fusion" flow that uses SQL for behavioral narrowing, Vector Search for semantic refinement, and SQL for profile enrichment.

---

## 2. High-Level Architecture

```
                   ┌──────────────────────┐
                   │     CRM Data         │
                   │  (PostgreSQL)        │
                   └──────────┬───────────┘
                               │
                   ┌──────────▼───────────┐
                   │   Clickstream Data   │
                   │   (Event Layer)      │
                   └──────────┬───────────┘
                               │
                      Data Ingestion Pipeline (Python)
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
  ┌───────▼────────┐                       ┌───────▼────────┐
  │ Relational DB  │                       │  Vector Store  │
  │ (Postgres)     │                       │  (Qdrant)      │
  └───────┬────────┘                       └───────┬────────┘
          │                                        │
          └───────────────┬────────────────────────┘
                          │
                 Orchestration Layer (LlamaIndex)
                          │
                 Agentic Reasoning (Gemini 2.0 Flash)
                          │
                   Interactive CLI
```

---

## 3. The "True Hybrid Fusion" Strategy

Unlike simple hybrid search (vector + metadata), this system implements a multi-stage orchestration flow:

1. **SQL Gate (Narrowing)**: Filters the 500-customer population based on hard behavioral event conditions (e.g., "purchased red socks in 60 days").
2. **Vector Refinement (Semantic)**: Performs a semantic search *inside* the filtered subset to match subjective intent (e.g., "interested in luxury").
3. **SQL Enrichment (Enrichment)**: Retrieves the full, campaign-ready customer profiles (email, first name, etc.) for the resulting IDs.

---

## 4. Behavioral Profiling

During ingestion, the system extracts two types of intelligence:

### 4.1 Calculated Interests
The system analyzes clickstream weights (Purchase: 3, Add-to-Cart: 2, View: 1) to derive:
- **Primary Interests**: Top product categories based on weighted behavior.
- **Preferred Colors**: Most interacted color palettes.

### 4.2 Luxury Tagging (Rule-Based Semantic)
Customers with a `total_spent` > 800 are tagged with: *"This customer likes luxury items."* This tag is embedded into the vector representation, enabling high-precision retrieval for luxury-oriented queries.

---

## 5. Technical Stack

- **Orchestration**: [LlamaIndex](https://www.llamaindex.ai/)  
  *Purpose*: LlamaIndex serves as the central **data framework** for our LLM application. It manages the connection between our raw data sources (Postgres, Qdrant) and the agent. Specifically, it handles the "Text-to-SQL" translation, vector store abstraction, and provides the `ReActAgent` loop that orchestrates multi-step tool usage.
- **Vector DB**: [Qdrant](https://qdrant.tech/) (Distance: Cosine)
- **Relational DB**: [PostgreSQL](https://www.postgresql.org/)
- **LLM**: Gemini 2.0 Flash (via [OpenRouter](https://openrouter.ai/))
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence-Transformers)

---

## 6. Tool Logic & Descriptions

The system exposes three primary tools to the agent, each with specific logic:

### 6.1 `sql_analytics`
- **Purpose**: Deterministic analysis and counts.
- **Logic**: Uses LlamaIndex's `NLSQLTableQueryEngine` to translate natural language into PostgreSQL queries. 
- **Best for**: "How many...?", "What is the average...?", "Give me a count of...".

### 6.2 `hybrid_discovery` (The Architectural Core)
- **Purpose**: Combined behavioral + semantic search.
- **Logic**:
  1. **SQL Gate**: Runs a `SELECT DISTINCT customer_id` based on `sql_where` (behavioral events).
  2. **Metadata Filter**: Creates an `IN` filter for the resulting IDs.
  3. **Vector Search**: Executes a similarity search on the semantic profiles *within* that ID set.
- **Best for**: "Customers who bought X and are interested in Y."

### 6.3 `sql_data_retriever`
- **Purpose**: Raw data retrieval for profile enrichment.
- **Logic**: Executes direct SQL `SELECT` queries to fetch the full JSON records for specific customer IDs.
- **Best for**: "Show me the details...", "Return as JSON".

---

## 7. Data Model Examples

### 6.1 Vector Payload
This is how a customer is represented in the vector store:

```json
{
  "customer_id": 1,
  "text": "Customer User1 Test1 (user1@demo.com) from PL, age 19, favorite color red. Total spent: 730.19. Primary interests: socks, shoes. Preferred colors: black, green.",
  "metadata": {
    "customer_id": 1,
    "first_name": "User1",
    "last_name": "Test1",
    "email": "user1@demo.com",
    "country": "PL",
    "age": 19,
    "total_spent": 730.19,
    "favorite_color": "red",
    "calculated_interests": "Primary interests: socks, shoes. Preferred colors: black, green.",
    "likes_luxury": false
  }
}
```

---

## 7. Operational Features

### 7.1 Cost Observability (NFR1)
The engine includes a `TokenCountingHandler` that logs real-time usage for every discovery query:
- LLM Prompt/Completion tokens
- Embedding tokens
- Transactional cost estimation parity

### 7.2 Tool Debugging
Enable transparency by monitoring tool arguments in real-time:
- View exact SQL queries generated by the agent.
- View metadata filter dictionaries applied to vector search.

---

## 8. How to Run

1. **Bootstrap**: `docker-compose up -d`
2. **Ingest**: `python ingest_data.py`
3. **Discover**: `python main.py`

**Sample Discoveries:**
- *"Suggest customers for a luxury red-themed fashion campaign."*
- *"Find users who bought socks and are interested in high-end lifestyle."*