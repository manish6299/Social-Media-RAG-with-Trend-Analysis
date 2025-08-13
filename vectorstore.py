import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import get_embeddings_model
import chromadb

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

def create_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)

def index_documents(docs):
    embeddings = get_embeddings_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    client = create_chroma_client()
    collection = client.get_or_create_collection("social_media_rag")

    for d in docs:
        if not all(k in d for k in ("id", "text", "created_at", "source")):
            raise ValueError(f"Missing required keys in document: {d}")

        chunks = text_splitter.split_text(d["text"])
        metadatas = [
            {
                "source": d["source"],
                "created_at": d["created_at"],
                **d.get("meta", {})
            }
            for _ in chunks
        ]
        ids = [f"{d['id']}_chunk_{i}" for i in range(len(chunks))]
        
        # Get embeddings for all chunks at once
        emb_vectors = embeddings.embed_documents(chunks)

        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
            embeddings=emb_vectors
        )

    print("✅ Indexing complete")

def query_vectorstore(query_text, top_k=5):
    embeddings = get_embeddings_model()
    client = create_chroma_client()

    try:
        collection = client.get_collection(name="social_media_rag")
    except Exception:
        collection = client.create_collection(name="social_media_rag")

    q_emb = embeddings.embed_query(query_text)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results