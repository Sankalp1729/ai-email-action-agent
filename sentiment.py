from functools import lru_cache

from transformers import pipeline


@lru_cache(maxsize=1)
def _get_sentiment_analyzer():
    try:
        return pipeline("sentiment-analysis")
    except Exception:
        return None


def _fallback_sentiment(text):
    lowered = (text or "").lower()
    positive_words = ["great", "good", "thanks", "appreciate", "awesome", "happy", "love", "resolved"]
    negative_words = ["urgent", "issue", "failed", "blocked", "delay", "error", "problem", "angry", "escalation"]

    positive_hits = sum(word in lowered for word in positive_words)
    negative_hits = sum(word in lowered for word in negative_words)

    if negative_hits > positive_hits:
        return "NEGATIVE", 0.74
    if positive_hits > negative_hits:
        return "POSITIVE", 0.74
    return "NEUTRAL", 0.5


def analyze_sentiment(text):
    sentiment_analyzer = _get_sentiment_analyzer()
    if sentiment_analyzer is None:
        return _fallback_sentiment(text)

    try:
        result = sentiment_analyzer(text)[0]
        return result["label"], result["score"]
    except Exception:
        return _fallback_sentiment(text)
