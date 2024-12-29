from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# Remove the hardcoded API key and use environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document(BaseModel):
    content: str
    metadata: dict = {}

class SearchQuery(BaseModel):
    query: str
    k: int = 5

class SearchResponse(BaseModel):
    documents: List[Document]
    scores: List[float]

# Global vectorstore
vectorstore = None

@app.on_event("startup")
async def startup_event():
    """Initialize the vectorstore on startup"""
    try:
        await initialize_vectorstore()
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore: {e}")
        raise

async def initialize_vectorstore():
    """Initialize the vectorstore with documents"""
    global vectorstore
    
    # Get PDF documents
    pdf_dir = Path("rag_docs")
    pdf_dir.mkdir(exist_ok=True)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found")
        return
    
    # Load and process documents
    docs = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading {pdf}: {e}")
    
    if not docs:
        logger.warning("No documents loaded")
        return
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = splitter.split_documents(docs)
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        collection_name="mental-health-docs"
    )
    logger.info("Vectorstore initialized successfully")

@app.get("/health")
async def health_check():
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not initialized")
    return {"status": "healthy"}

@app.post("/search", response_model=SearchResponse)
async def search_documents(query: SearchQuery):
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not initialized")
    
    try:
        docs_and_scores = vectorstore.similarity_search_with_score(
            query.query,
            k=query.k
        )
        
        return SearchResponse(
            documents=[
                Document(
                    content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc, _ in docs_and_scores
            ],
            scores=[score for _, score in docs_and_scores]
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 