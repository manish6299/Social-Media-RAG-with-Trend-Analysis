# trend_analyzer.py
from collections import Counter, defaultdict
import re
import math
import time
from datetime import datetime, timedelta
import pandas as pd

def extract_terms(text):
    tokens = re.findall(r"#\w+|\w+", text.lower())
    tokens = [t for t in tokens if len(t) > 2]
    return tokens

def compute_term_frequencies(docs):
    counts = Counter()
    for d in docs:
        counts.update(extract_terms(d["text"]))
    return counts

def detect_trends(recent_docs, older_docs, top_n=20):
    recent_counts = compute_term_frequencies(recent_docs)
    older_counts = compute_term_frequencies(older_docs)

    trend_scores = []
    for term, r_count in recent_counts.items():
        o_count = older_counts.get(term, 0.1)  
        score = (r_count / o_count) * math.log(1 + r_count)
        trend_scores.append((term, score, r_count, o_count))
    trend_scores.sort(key=lambda x: x[1], reverse=True)
    return trend_scores[:top_n]


if __name__ == "__main__":
    pass
