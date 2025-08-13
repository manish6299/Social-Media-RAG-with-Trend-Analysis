#embeddings.py

import os
from typing import List
from dotenv import load_dotenv

# Try to import sentence_transformers (for local embeddings)
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False

load_dotenv()

class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

def get_embeddings_model():

    if not LOCAL_EMBEDDINGS_AVAILABLE:
        raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    print("Using LOCAL embeddings")
    return LocalEmbeddings()

if __name__ == "__main__":
    print("Testing Local Embeddings...")
    try:
        emb = get_embeddings_model()
        test_embedding = emb.embed_query("Test query")
        print(f"Success! Embedding length: {len(test_embedding)}")
    except Exception as e:
        print(f"Failed: {str(e)}")