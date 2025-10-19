from mcp.server.fastmcp import FastMCP, Context
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import os

# Custom embedding function class for SentenceTransformer
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

@dataclass
class AppContext:
    vectorstore: Chroma
    collection_name: str

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize Chroma vector store with SentenceTransformer and manage lifecycle"""
    # Get environment variables or use defaults
    persistence_directory = os.environ.get("CHROMA_PERSISTENCE_DIRECTORY", "./chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION", "vue3_components_docs")
    
    vectorstore = None
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_function = SentenceTransformerEmbeddings(embedder)
    
    # Initialize Chroma vector store
    vectorstore = Chroma(
        persist_directory=persistence_directory,
        embedding_function=embedding_function,
        collection_name=collection_name
    )
    
    
    try:
        yield AppContext(vectorstore=vectorstore, collection_name=collection_name)
    finally:
        pass

# Initialize the MCP server with lifespan management
mcp = FastMCP("Vue3 Components Documentation Search", 
              lifespan=app_lifespan, 
              dependencies=["langchain", "sentence-transformers"])

@mcp.tool()
async def search_documents(query: str, limit: int = 30, ctx: Context = None) -> str:
    """
    Search for Vue3 component documentation using semantic similarity.
    
    Args:
        query: The search query text about Vue3 components
        limit: Maximum number of results to return (default: 30)
    
    Returns:
        Matching documentation entries as text
    """
    if ctx is None:
        return "Error: Context not available"
    
    # Access the vectorstore from context
    vectorstore = ctx.request_context.lifespan_context.vectorstore
    
    
    # Use similarity search with scores
    results = vectorstore.similarity_search_with_score(query, k=limit)
    
    # Format the results
    if not results:
        return "No matching documents found."
    
    formatted_results = []
    for i, (doc, score) in enumerate(results):
        result = f"--- Document {i+1} (Similarity Score: {score:.4f}) ---\n"
        
        # Add metadata if available
        if doc.metadata:
            result += "Metadata:\n"
            for key, value in doc.metadata.items():
                result += f"  {key}: {value}\n"
        
        # Add document content
        result += f"\nContent:\n{doc.page_content}\n\n"
        formatted_results.append(result)
    
    summary = f"\nReturned {len(results)} matching documents for query: '{query}'"
    return "\n".join(formatted_results) + summary

if __name__ == "__main__":
    mcp.run(transport='stdio')
    mcp.run(transport='stdio')
    print(2331222)
