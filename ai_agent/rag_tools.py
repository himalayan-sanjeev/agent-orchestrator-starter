from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import Tool
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

def search_docs(query):
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

search_tool = Tool(
    name="DocumentSearch",
    description="Search the vector database for document knowledge",
    func=search_docs
)
