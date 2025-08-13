import os
from typing import List
from dotenv import load_dotenv
import requests

load_dotenv()

class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise Exception(f"HuggingFace API Error: {response.text}")
        return response.json()

def get_embeddings_model():
    # First try using HuggingFace API
    if os.getenv('HF_API_KEY'):
        print("Using HuggingFace API embeddings")
        return HuggingFaceEmbeddings()
    
    # Fallback to Chroma's default embeddings if no API key
    try:
        from chromadb.utils import embedding_functions
        print("Using Chroma's default embeddings as fallback")
        return embedding_functions.DefaultEmbeddingFunction()
    except ImportError:
        raise ImportError("Neither HuggingFace API nor Chroma embeddings available. Please set HF_API_KEY")

if __name__ == "__main__":
    print("Testing Embeddings...")
    try:
        emb = get_embeddings_model()
        test_embedding = emb.embed_query("Test query")
        print(f"Success! Embedding length: {len(test_embedding)}")
    except Exception as e:
        print(f"Failed: {str(e)}")
