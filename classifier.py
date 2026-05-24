from functools import lru_cache

from transformers import pipeline

# Define candidate labels (actions)
labels = ["reply", "forward", "mark as important", "delete", "ignore"]

MODEL_NAME = "valhalla/distilbart-mnli-12-1"


@lru_cache(maxsize=1)
def _get_classifier():
    try:
        return pipeline("zero-shot-classification", model=MODEL_NAME)
    except Exception:
        return None


def _fallback_classify(message: str):
    text = (message or "").lower()

    rules = [
        ("mark as important", ["urgent", "asap", "deadline", "important", "review", "blocked", "escalation", "action required"]),
        ("reply", ["please", "can you", "could you", "need your", "confirm", "share", "let me know", "respond"]),
        ("forward", ["fyi", "please forward", "shared with", "include", "loop in"]),
        ("delete", ["unsubscribe", "promo", "sale", "marketing", "advertisement", "spam"]),
        ("ignore", ["newsletter", "update", "notification", "reminder"]),
    ]

    for action, keywords in rules:
        if any(keyword in text for keyword in keywords):
            confidence = 0.78 if action in {"mark as important", "reply"} else 0.66
            return action, confidence

    return "reply", 0.51


def classify_message(message: str):
    """Classify a message and return the best action and its confidence."""
    classifier = _get_classifier()
    if classifier is None:
        return _fallback_classify(message)

    try:
        result = classifier(message, labels)
        top_action = result["labels"][0]
        top_confidence = result["scores"][0]
        return top_action, top_confidence
    except Exception:
        return _fallback_classify(message)
