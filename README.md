# Agentic RAG Chatbot 🤖🚀
A local, intelligent Agentic Retrieval-Augmented Generation (RAG) system built with **LangChain**, **LangGraph**, and **ChromaDB**. 
It indexes documents locally and uses an LLM Agent (via **OpenRouter**) to decide dynamically when to query the local knowledge base or answer using its general reasoning.
## Features
* 📁 **Automated Ingestion**: Reads all `.txt` documents inside the `docs/` folder.
* 🧠 **Local Embeddings**: Uses Ollama's `nomic-embed-text` to generate embeddings.
* 💾 **Vector Store**: Persists chunks in a local `Chroma` database.
* 🤖 **ReAct Agent**: Wraps the database retrieval as a tool, allowing the agent to decide when to search the documents and when to reason directly.
---
## Setup Instructions
### 1. Clone & Navigate
```bash
git clone <your-repo-url>
cd chatbot
2. Configure Environment Variables
Create a .env file in the root directory:

env


OPENROUTER_API_KEY=your_openrouter_api_key_here
3. Install Dependencies
Set up your virtual environment and install the required packages:

bash


python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
How to Run
Step 1: Ingest Documents
Place your text files (e.g. SpaceX.txt, test.txt) inside the docs/ directory, then run:

bash


python 1.ingestion.py
This will chunk, embed, and store the documents in db/chroma_db/.

Step 2: Run the Agent
To start the chatbot and ask questions:

bash

python agent.py