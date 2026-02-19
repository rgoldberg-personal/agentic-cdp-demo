import asyncio
import time
from orchestration import run_query_async, token_counter

async def run_benchmark():
    test_queries = [
        "How many customers are in the database?",
        "Find blue-themed customers from Germany who spent more than 500.",
        "Suggest a list of customers interested in jacket and shoes.",
        "Find customers who bought red socks in the last 6 months and like luxury. Show them as JSON.",
        "Recommend an audience for a luxury high-end fashion campaign in Poland."
    ]
    
    print("\n" + "="*60)
    print("      AI Audience Discovery Platform - Benchmark Evaluation")
    print("="*60 + "\n")
    
    total_tokens = 0
    start_time = time.time()
    
    for i, q in enumerate(test_queries, 1):
        print(f"Scenario {i}: {q}")
        print("-" * 30)
        
        # We don't want to print all debug info here to keep repo clean
        # but orchestration.py will still print it. That's fine.
        
        token_counter.reset_counts()
        response = await run_query_async(q)
        
        usage = token_counter.total_llm_token_count
        total_tokens += usage
        
        print(f"Result Length: {len(response)} chars")
        print(f"Tokens Used: {usage}")
        print("-" * 30 + "\n")
        
    duration = time.time() - start_time
    print("="*60)
    print("      Benchmark Summary")
    print("="*60)
    print(f"Total Scenarios: {len(test_queries)}")
    print(f"Total Processing Time: {duration:.2f} seconds")
    print(f"Total Token Consumption: {total_tokens}")
    print(f"Average Tokens per Query: {total_tokens / len(test_queries):.0f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
