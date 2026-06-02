"""
migrate_to_pinecone.py - Script to upload knowledge base documents to Pinecone
=============================================================================
Before running this script, ensure you have installed the required libraries:
    pip install langchain-pinecone pinecone-client

Make sure your .env file contains:
    PINECONE_API_KEY=your-api-key
    PINECONE_INDEX_NAME=your-index-name
    GOOGLE_API_KEY=your-google-api-key
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"

def main():
    # 1. Validate Environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not pinecone_api_key or not pinecone_index_name:
        print("Error: PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in your .env file.")
        sys.exit(1)
        
    if not google_api_key:
        print("Error: GOOGLE_API_KEY must be set in your .env file to generate embeddings.")
        sys.exit(1)

    print("Step 1: Initializing Google Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)

    # 2. Load Documents
    print("Step 2: Loading documents from knowledge base...")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Error: Knowledge base directory not found at {KNOWLEDGE_BASE_DIR}")
        sys.exit(1)
        
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR, glob="**/*.txt",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    
    # Tag documents with appropriate metadata
    for doc in docs:
        doc.metadata["user_id"] = "public"
        doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "Internal"))
        
    print(f"   Loaded {len(docs)} documents.")

    # 3. Chunk Documents
    print("Step 3: Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    # Filter out empty documents
    safe_docs = [d for d in docs if d.page_content is not None]
    chunks = splitter.split_documents(safe_docs)
    print(f"   Split into {len(chunks)} chunks.")

    # 4. Upload to Pinecone
    print(f"Step 4: Connecting to Pinecone index: '{pinecone_index_name}' and uploading chunks...")
    try:
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=pinecone_index_name,
            pinecone_api_key=pinecone_api_key
        )
        print("Success! Upload to Pinecone completed successfully.")
        print(f"Uploaded {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error uploading to Pinecone: {str(e)}")
        print("\nNote: Make sure your Pinecone index is configured with 3072 dimensions (matching models/gemini-embedding-001) and 'cosine' distance metric.")
        sys.exit(1)

if __name__ == "__main__":
    main()
