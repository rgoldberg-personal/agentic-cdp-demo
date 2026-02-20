import sys
import argparse
import asyncio

def show_welcome():
    print("\n" + "="*60)
    print("      🚀 AGENTIC CDP: AUDIENCE DISCOVERY PLATFORM")
    print("="*60 + "\n")
    print("Usage: python3 main.py [command]")
    print("Commands:")
    print("  cli        Launch the interactive chat console")
    print("  ingest     Run the data ingestion pipeline")
    print("  benchmark  Run the system performance benchmark")
    print("-" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Agentic CDP Platform")
    parser.add_argument("command", nargs="?", help="Command to run (cli, ingest, benchmark)")
    
    args = parser.parse_args()
    
    if args.command == "cli":
        from cli import start_console
        asyncio.run(start_console())
    elif args.command == "ingest":
        from ingest import run_ingestion
        run_ingestion()
    elif args.command == "benchmark":
        from benchmark import run_benchmark
        asyncio.run(run_benchmark())
    else:
        show_welcome()

if __name__ == "__main__":
    main()
