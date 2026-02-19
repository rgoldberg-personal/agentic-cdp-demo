import asyncio
import sys
from orchestration import run_query_async

async def main():
    print("\n" + "="*50)
    print("      AI Audience Discovery Platform CLI")
    print("="*50)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Ask a question about your audience: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            if not query.strip():
                continue

            print("\nThinking...")
            response = await run_query_async(query)
            print(f"\nAI Response:\n{response}\n")
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    print("\nGoodbye!")
