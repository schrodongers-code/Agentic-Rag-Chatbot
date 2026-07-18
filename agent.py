import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

# Load environment variables (.env)
load_dotenv()

def main():
    print("=== Initializing Agentic RAG ===\n")
    
    # 1. Load Chroma DB and setup retriever
    persistent_directory = "db/chroma_db"
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # 2. Wrap the retriever in a Tool
    # The description helps the LLM decide when to call the tool
    retriever_tool = create_retriever_tool(retriever,name="company_knowledge_search",
                                           description="Searches and retrieves general documents, test records, SpaceX details, and other local text information.")
    tools = [retriever_tool]
    
    # 3. Initialize the LLM via OpenRouter
    # We use Google's Gemini 2.5 Flash as it is fast, cheap, and supports tool calling
    llm = ChatOpenAI( model="google/gemma-4-26b-a4b-it:free", 
                       api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1" )
    
    # 4. Create the Agent
    agent = create_agent(llm, tools)
    
    # --- Example Executions ---

    # Query 1: Requires SpaceX knowledge (Agent will use the tool)
    query_1 = input("Enter your Query") 
    print(f"User Query 1: '{query_1}'")
    response_1 = agent.invoke({"messages": [("user", query_1)]})
    print(f"Agent Reply:\n{response_1['messages'][-1].content}\n")
    
    print("-" * 60)
    
    # Query 2: General request (Agent will answer directly WITHOUT using the tool)
    query_2 = "Write a short 2-line poem about space."
    print(f"User Query 2: '{query_2}'")
    response_2 = agent.invoke({"messages": [("user", query_2)]})
    print(f"Agent Reply:\n{response_2['messages'][-1].content}\n")

if __name__ == "__main__":
    main()