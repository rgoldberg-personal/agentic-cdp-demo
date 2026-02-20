import asyncio
import sys
import os
from agent import chat_async, reset_chat

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

async def start_console():
    os.system('clear' if os.name == 'nt' else 'clear')
    print(f"{Colors.BOLD}{Colors.HEADER}" + "="*60)
    print("      🚀 AGENTIC CDP: AUDIENCE DISCOVERY CONSOLE")
    print("="*60 + f"{Colors.ENDC}")
    print(f"{Colors.CYAN}Commands: 'exit' to quit | 'clear' to reset history | 'help' for schema{Colors.ENDC}\n")

    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.BLUE}You > {Colors.ENDC}")
            
            if not user_input.strip():
                continue
                
            cmd = user_input.lower().strip()
            if cmd in ['exit', 'quit', 'bye']:
                print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}")
                break
            
            if cmd == 'clear':
                reset_chat()
                print(f"\n{Colors.WARNING}[SYSTEM] Chat history cleared.{Colors.ENDC}\n")
                continue
                
            if cmd == 'help':
                print(f"\n{Colors.BOLD}Schema Info:{Colors.ENDC}")
                print("- events: customer_id, event_type, product, color, price, event_timestamp")
                print("- customers: customer_id, first_name, last_name, email, country, age, total_spent\n")
                continue

            print(f"\n{Colors.CYAN}Thinking...{Colors.ENDC}")
            response = await chat_async(user_input)
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}Agent >{Colors.ENDC}")
            print(response)
            print(f"\n{Colors.CYAN}" + "-"*60 + f"{Colors.ENDC}\n")
            
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Interrupted. Type 'exit' to quit.{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.FAIL}Error: {str(e)}{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        asyncio.run(start_console())
    except KeyboardInterrupt:
        pass
