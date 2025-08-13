# evaluation.py
import time
from rag_system import generate_answer

def measure_latency(query):
    t0 = time.time()
    _ = generate_answer(query)
    return time.time() - t0

# For retrieval accuracy you'd need test queries and expected doc ids/labels.
def simple_demo():
    queries = ["What are people saying about climate strikes?"]
    for q in queries:
        latency = measure_latency(q)
        print(f"Query latency: {latency:.2f}s")

if __name__ == "__main__":
    simple_demo()
