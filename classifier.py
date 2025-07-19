from transformers import pipeline

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels (actions)
labels = ["reply", "forward", "mark as important", "delete", "ignore"]

def classify_message(message: str):
    """
    Classify a message and return the best action and its confidence.
    """
    result = classifier(message, labels)
    top_action = result["labels"][0]
    top_confidence = result["scores"][0]
    return top_action, top_confidence
