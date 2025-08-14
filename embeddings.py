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
        # For sentence-transformers models, use 'inputs' with proper format
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        
        # Retry if model is still loading
        for attempt in range(5):
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, list):
                    # If data is a list of embeddings
                    if len(data) > 0 and isinstance(data[0], list):
                        return data
                    # If data is a single embedding
                    elif len(data) > 0 and isinstance(data[0], (int, float)):
                        return [data]
                
                # If data is a single embedding (not in a list)
                if isinstance(data, dict) and 'embeddings' in data:
                    return data['embeddings']
                
                # Fallback - try to return as is
                return data if isinstance(data, list) else [data]
                
            elif response.status_code == 503:  # Model loading
                print(f"Model is loading... retrying in 5 seconds (Attempt {attempt+1}/5)")
                time.sleep(5)
            else:
                print(f"API Response: {response.text}")
                print(f"Status Code: {response.status_code}")
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
        print(f"First few values: {test_embedding[:5]}")
    except Exception as e:
        print(f"Failed: {str(e)}")
