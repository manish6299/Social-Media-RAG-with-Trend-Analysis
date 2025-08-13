import streamlit as st
from rag_system import generate_answer
from trend_analyzer import detect_trends
from social_media_ingestor import fetch_tweets, fetch_reddit
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from vectorstore import query_vectorstore


st.set_page_config(page_title="Social Media RAG Demo", layout="wide")
st.title("Social Media RAG with Trend Analysis")

# Sidebar controls
# Sidebar design
with st.sidebar:
    st.markdown("## ⚙️ RAG & Trend Analysis Settings")
    st.markdown("---")

    # Retrieval settings
    st.markdown("**📄 Retrieval Settings**")
    top_k = st.slider(
        "Number of documents to retrieve (top_k)", 
        min_value=1, max_value=10, value=5, step=1, 
        help="Number of top relevant documents to use for generating the answer."
    )

    # Trend analysis time window
    st.markdown("**⏳ Trend Analysis Window**")
    time_window = st.selectbox(
        "Select time window", 
        ["24h", "48h", "1 week"], 
        help="Set how far back in time to analyze social media trends."
    )

    # Platform selection
    st.markdown("**🌐 Social Media Platforms**")
    platforms = st.multiselect(
        "Select platforms to analyze", 
        ["Twitter", "Reddit"], 
        default=["Twitter", "Reddit"], 
        help="Choose one or more social media sources for trend analysis."
    )

    # Trend filtering
    st.markdown("**📈 Trend Filters**")
    min_trend_score = st.slider(
        "Minimum Trend Score", 
        min_value=1, max_value=100, value=10, step=1, 
        help="Set the minimum trend score threshold to consider a topic significant."
    )

    min_term_length = st.slider(
        "Minimum Term Length", 
        min_value=2, max_value=10, value=3, step=1, 
        help="Minimum number of characters in a keyword/term."
    )

    st.markdown("---")
    st.caption("Adjust settings to customize trend detection and retrieval behavior.")

# Modified extract_terms function with min_length parameter
def extract_terms(text, min_length=3):
    tokens = re.findall(r"#\w+|\w+", text.lower())
    tokens = [t for t in tokens if len(t) >= min_length and not t.startswith(('http', '@'))]
    return tokens

# Main tabs
tab1, tab2 = st.tabs(["Search", "Trend Dashboard"])

with tab1:
    query = st.text_input("Ask about a trending topic or event", value="What are people saying about X?")
    
    if st.button("Search"):
        with st.spinner("Retrieving and generating..."):
            res = generate_answer(query, top_k=top_k)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Answer")
                st.write(res["answer"])
                
            with col2:
                st.metric("Retrieved Contexts", len(res["context_snippets"]))
                # Fix for NaTType error
                dates = [m["created_at"] for m in res["metadatas"] if m.get("created_at")]
                if dates:
                    try:
                        avg_date = pd.to_datetime(dates).mean()
                        st.metric("Avg. Post Date", avg_date.strftime("%Y-%m-%d"))
                    except:
                        st.metric("Avg. Post Date", "N/A")
                else:
                    st.metric("Avg. Post Date", "No dates found")
            
            st.subheader("Context snippets (retrieved)")
            for i, snippet in enumerate(res["context_snippets"]):
                with st.expander(f"Snippet {i+1} | Source: {res['metadatas'][i].get('source', 'unknown')}"):
                    st.text(snippet)
                    st.caption(f"Created: {res['metadatas'][i].get('created_at', 'unknown')} | Score: {res['metadatas'][i].get('score', 'N/A')}")
        

with tab2:
    st.header("Trend Analysis Dashboard")
    
    if st.button("Refresh Trends"):
        with st.spinner("Analyzing trends..."):
            # Simulate time windows
            recent_docs = []
            older_docs = []
            
            if "Twitter" in platforms:
                with st.spinner("Fetching recent tweets..."):
                    recent_docs.extend(fetch_tweets("lang:en", max_results=100))
                with st.spinner("Fetching older tweets..."):
                    older_docs.extend(fetch_tweets("lang:en", max_results=50))
            
            if "Reddit" in platforms:
                with st.spinner("Fetching recent Reddit posts..."):
                    recent_docs.extend(fetch_reddit("all", limit=100))
                with st.spinner("Fetching older Reddit posts..."):
                    older_docs.extend(fetch_reddit("all", limit=50))
            
            # Run trend analysis
            trends = detect_trends(recent_docs, older_docs, top_n=20)
            
            # Filter by score and term length
            trends = [t for t in trends if t[1] >= min_trend_score and len(t[0]) >= min_term_length]
            
            if not trends:
                st.warning("No significant trends found. Try adjusting the filters.")
            else:
                # Display trends
                st.subheader("Top Trending Terms")
                df = pd.DataFrame(trends, columns=["Term", "Trend Score", "Recent Count", "Older Count"])
                df["Growth"] = (df["Recent Count"] / df["Older Count"]).round(1)
                
                # Sort by trend score
                df = df.sort_values("Trend Score", ascending=False)
                
                # Show top trends
                st.dataframe(df.head(10))
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                df.head(10).plot.barh(x="Term", y="Trend Score", ax=ax)
                ax.set_title("Top Trending Terms by Score")
                st.pyplot(fig)
                
                # Term frequency over time
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                df.head(5).set_index("Term")[["Recent Count", "Older Count"]].plot.bar(ax=ax2)
                ax2.set_title("Term Frequency Comparison")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)
                
                # Show sample posts for selected trend
                selected_trend = st.selectbox("View posts for trend:", df["Term"].tolist())
                if selected_trend:
                    st.subheader(f"Sample posts about '{selected_trend}'")
                    sample_posts = [d for d in recent_docs if selected_trend.lower() in d["text"].lower()][:5]
                    
                    if not sample_posts:
                        st.info("No sample posts found for this term")
                    else:
                        for post in sample_posts:
                            with st.expander(f"{post.get('source', 'unknown')} - {post.get('created_at', 'unknown date')}"):
                                st.write(post["text"])
                                if post.get('meta'):
                                    st.caption(f"Engagement: {post['meta'].get('score', post['meta'].get('public_metrics', {}))}")
                