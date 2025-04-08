from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from crew_agents import run_crew_for

load_dotenv()

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

# Vector DB (persistent)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Retriever setup
retriever = db.as_retriever(search_kwargs={"k": 3})

# Gemini Generative Model setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# RAG QA chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

@app.get("/api/rag_query")
async def rag_query(q: str):
    response = qa_chain.run(q)
    return {"query": q, "answer": response.strip()}


@app.get("/api/crew_summary")
async def crew_summary(topic: str):
    final_output = run_crew_for(topic)
    return {"topic": topic, "summary": final_output}
