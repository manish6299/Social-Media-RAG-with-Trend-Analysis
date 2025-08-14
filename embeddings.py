import os
import time
from typing import List
from dotenv import load_dotenv
import requests

load_dotenv()

class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Hugging Face SentenceTransformer API accepts either a single string or a list of strings
        payload = {
            "inputs": texts if len(texts) > 1 else texts[0],
            "options": {"wait_for_model": True}
        }

        # Retry if model is still loading
        for attempt in range(5):
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Ensure output is a list of lists
                return data if isinstance(data[0], list) else [data]
            elif response.status_code == 503:  # Model loading
                print(f"Model is loading... retrying in 5 seconds (Attempt {attempt+1}/5)")
                time.sleep(5)
            else:
                raise Exception(f"HuggingFace API Error: {response.text}")

        raise Exception("HuggingFace model failed to load after multiple attempts.")

def get_embeddings_model():
    if os.getenv('HF_API_KEY'):
        print("Using HuggingFace API embeddings")
        return HuggingFaceEmbeddings()
    
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
