import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def test_llm():
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )
    
    response = llm.invoke("Hello, who are you?")
    print(response.content)

if __name__ == "__main__":
    test_llm()
