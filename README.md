# Social Media RAG with Trend Analysis

A comprehensive Retrieval-Augmented Generation (RAG) system that analyzes social media trends from Twitter and Reddit, providing intelligent insights using AI-powered search and analysis.

## 🌟 Features

- **Multi-Platform Data Ingestion**: Fetch real-time data from Twitter and Reddit
- **Vector-Based Search**: ChromaDB-powered semantic search with local embeddings
- **AI-Powered Analysis**: Groq LLM integration for intelligent trend interpretation
- **Interactive Dashboard**: Streamlit-based web interface with customizable settings
- **Trend Detection**: Advanced algorithms to identify emerging and viral topics
- **Real-time Visualization**: Charts and graphs showing trend patterns and growth
- **Contextual Responses**: RAG system provides source-backed answers with citations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Vector Store  │    │   AI Analysis   │
│                 │    │                 │    │                 │
│ • Twitter API   │───▶│ • ChromaDB     │───▶│ • Groq LLM      │
│ • Reddit API    │    │ • Embeddings   │    │ • Trend Analysis│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │ Streamlit UI    │
                        │                 │
                        │ • Search Tab    │
                        │ • Trend Dashboard│
                        │ • Visualizations│
                        └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- Twitter Developer Account (for Bearer Token)
- Reddit Developer Account (for API credentials)
- Groq API Key

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd social-media-rag
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements.txt contents:**
```
streamlit
tweepy
praw
requests
langchain
langchain-groq
chromadb
sentence-transformers
python-dotenv
pandas
numpy
matplotlib
scikit-learn
nltk
transformers
torch
ragas
tqdm
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory with the following structure:

```env
# Twitter API
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=your_app_name:version (by /u/your_username)

# Groq API
GROQ_API_KEY=your_groq_api_key_here

# ChromaDB Directory
CHROMA_DB_DIR=./chroma_db
```

## 🔑 API Setup Guide

### Twitter API Setup

1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a new project/app
3. Generate Bearer Token
4. Copy the Bearer Token to your `.env` file

### Reddit API Setup

1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" type
4. Note down the Client ID (under the app name) and Client Secret
5. Add credentials to your `.env` file

### Groq API Setup

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up/Login and create an API key
3. Copy the API key to your `.env` file

## 📁 Project Structure

```
social-media-rag/
│
├── app.py                    # Main Streamlit application
├── rag_system.py            # RAG implementation with Groq
├── social_media_ingestor.py # Twitter & Reddit data fetching
├── trend_analyzer.py        # Trend detection algorithms
├── vectorstore.py          # ChromaDB operations
├── embeddings.py           # Local embeddings handling
├── evaluation.py           # Performance evaluation
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── chroma_db/            # ChromaDB storage (auto-created)
└── README.md            # This file
```

## 🏃‍♂️ Running the Application

### Step 1: Initialize the Database

Before running the app for the first time, you may want to populate the database:

```bash
python social_media_ingestor.py
```

### Step 2: Start the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Step 3: Using the Interface

#### Search Tab
1. Enter your query about trending topics or events
2. Adjust retrieval settings in the sidebar
3. Click "Search" to get AI-powered insights

#### Trend Dashboard
1. Select platforms (Twitter/Reddit) to analyze
2. Set time windows and trend filters
3. Click "Refresh Trends" to see current trending topics
4. Explore visualizations and sample posts

## ⚙️ Configuration Options

### Sidebar Settings

- **Retrieval Settings**: Number of documents to retrieve (1-10)
- **Time Window**: Analysis period (24h, 48h, 1 week)
- **Platform Selection**: Choose Twitter, Reddit, or both
- **Trend Filters**: Minimum trend score and term length

### Advanced Configuration

Edit the following files for advanced customization:

- `embeddings.py`: Change embedding model
- `rag_system.py`: Modify LLM parameters
- `trend_analyzer.py`: Adjust trend detection algorithms

## 📊 Key Components

### RAG System (`rag_system.py`)
- Uses Groq's Llama-3.1-8b-instant model
- Retrieves relevant social media content
- Generates contextual answers with citations

### Data Ingestion (`social_media_ingestor.py`)
- Fetches recent tweets using Twitter API v2
- Collects Reddit posts from specified subreddits
- Standardizes data format across platforms

### Trend Analysis (`trend_analyzer.py`)
- Computes term frequencies and growth rates
- Identifies emerging and viral topics
- Calculates trend scores using logarithmic scaling

### Vector Store (`vectorstore.py`)
- Uses ChromaDB for persistent storage
- Implements semantic search capabilities
- Manages document chunking and indexing

## 🔍 Usage Examples

### Example Queries
- "What are people saying about climate change?"
- "Latest trends in artificial intelligence"
- "Public opinion on recent political events"
- "Viral memes and social movements"

### Sample Workflow
1. **Data Collection**: System fetches recent posts from social platforms
2. **Indexing**: Content is chunked and embedded into vector database
3. **Query Processing**: User questions are converted to embeddings
4. **Retrieval**: Most relevant content is identified and retrieved
5. **Generation**: AI generates comprehensive answers with source citations
6. **Trend Analysis**: System identifies and visualizes trending patterns

## 🧪 Testing and Evaluation

Run performance tests:

```bash
python evaluation.py
```

This will measure:
- Query response latency
- Retrieval accuracy (with test dataset)

## 📈 Performance Optimization

### Tips for Better Performance

1. **Embedding Model**: Default uses `all-MiniLM-L6-v2` for speed
2. **Chunking**: Adjust chunk size in `vectorstore.py` for your use case
3. **API Limits**: Be mindful of Twitter/Reddit API rate limits
4. **Database**: ChromaDB stores locally for persistence

### Scaling Considerations

- For production: Consider cloud-based vector databases
- For high volume: Implement batch processing
- For real-time: Add streaming data pipelines

## 🛠️ Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all keys are set in `.env`
2. **Rate Limits**: Twitter API has strict limits; implement backoff
3. **Memory Issues**: Reduce batch sizes for large datasets
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable verbose logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Streamlit](https://streamlit.io/) for the web interface
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- Twitter and Reddit APIs for data access

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed description

---

**Note**: This project is for educational and research purposes. Ensure compliance with platform terms of service and data privacy regulations when using social media APIs.
