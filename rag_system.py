#rag_system
import os
from embeddings import get_embeddings_model
from vectorstore import query_vectorstore
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Changed to Groq

load_dotenv()

def generate_answer(query: str, top_k=5):
    # 1) Retrieve documents (unchanged)
    res = query_vectorstore(query, top_k=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    # 2) Build context
    context = ""
    for i, doc_text in enumerate(docs):
        meta = metas[i]
        context += f"---\nSource: {meta.get('source')} | Created: {meta.get('created_at')}\n{doc_text}\n"

    prompt = (
    "You are an advanced social media trend analysis assistant that uses retrieved social media content "
    "to answer user queries with accurate, contextual, and timely insights.\n\n"
    
    "You must:\n"
    "1. Ingest and analyze posts from multiple platforms.\n"
    "2. Identify and highlight trending topics related to the query.\n"
    "3. Recognize and explain viral content patterns, memes, or hashtags.\n"
    "4. Provide cultural, social, or movement-related context if relevant.\n"
    "5. Indicate if the trend is emerging, peaking, or declining.\n"
    "6. Cite sources with both the platform/source name and creation timestamp (source + created_at).\n"
    "7. If information is speculative or unverified, clearly say so.\n\n"
    
    f"Retrieved Context:\n{context}\n\n"
    f"User Query:\n{query}\n\n"
    
    "Your output should be concise but information-rich, combining factual details with analytical insights. "
    "Include:\n"
    "- Summary of relevant content\n"
    "- Trend status and notable changes\n"
    "- Explanation of any viral/meme aspects\n"
    "- Social or cultural implications if applicable\n"
    "- Sources cited inline.\n"
)


    # 3) Use Groq instead of OpenRouter
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",  # Free model
        api_key=os.getenv("GROQ_API_KEY"),  # Add to .env
        temperature=0.7
    )
    
    answer = llm.invoke(prompt)
    return {"answer": answer.content, "context_snippets": docs, "metadatas": metas}