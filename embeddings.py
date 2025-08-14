class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Modified payload format to match what the API expects
        payload = {
            "inputs": {
                "source_sentence": texts[0] if len(texts) == 1 else None,
                "sentences": texts if len(texts) > 1 else None
            },
            "options": {"wait_for_model": True}
        }
        
        # Remove None values from payload
        if payload["inputs"]["source_sentence"] is None:
            del payload["inputs"]["source_sentence"]
        if payload["inputs"]["sentences"] is None:
            del payload["inputs"]["sentences"]

        # Retry if model is still loading
        for attempt in range(5):
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Handle different response formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "embedding" in data:
                    return [data["embedding"]]
                else:
                    return [data]
            elif response.status_code == 503:  # Model loading
                print(f"Model is loading... retrying in 5 seconds (Attempt {attempt+1}/5)")
                time.sleep(5)
            else:
                raise Exception(f"HuggingFace API Error: {response.text}")
        raise Exception("HuggingFace model failed to load after multiple attempts.")
