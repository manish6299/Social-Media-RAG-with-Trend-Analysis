# social_media_ingestor.py
import os
import time
import requests
import praw
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "media_analysis:0.1")

def fetch_tweets(query: str, max_results=100) -> List[Dict]:
    headers = {"Authorization": f"Bearer {TWITTER_BEARER}"}
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,lang,public_metrics,author_id",
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    docs = []
    for t in data:
        docs.append({
            "id": t["id"],
            "text": t["text"],
            "created_at": t.get("created_at"),
            "source": "twitter",
            "meta": {"public_metrics": t.get("public_metrics", {})}
        })
    return docs

def fetch_reddit(subreddit: str, limit=100) -> List[Dict]:
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)
    subreddit_obj = reddit.subreddit(subreddit)
    docs = []
    for post in subreddit_obj.hot(limit=limit):
        docs.append({
            "id": f"reddit_{post.id}",
            "text": post.title + "\n\n" + (post.selftext or ""),
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(post.created_utc)),
            "source": "reddit",
            "meta": {"score": post.score, "num_comments": post.num_comments, "subreddit": str(subreddit)}
        })
    return docs

if __name__ == "__main__":
    # Example usage
    tweets = fetch_tweets("climate change -is:retweet", max_results=50)
    posts = fetch_reddit("worldnews", limit=50)
    print(f"Fetched {len(tweets)} tweets and {len(posts)} reddit posts")
