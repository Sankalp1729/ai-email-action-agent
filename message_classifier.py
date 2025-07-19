from transformers import pipeline

# Load a zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible actions as classification labels
labels = ["reply", "forward", "mark as important", "delete", "ignore"]

def classify_message(message: str):
    """
    Classify the given email message into one of the predefined actions.

    Args:
        message (str): The email content to classify.

    Returns:
        tuple: (top_action, top_confidence) where:
            - top_action is the most appropriate action label.
            - top_confidence is the confidence score (float).
    """
    result = classifier(message, labels)
    top_action = result["labels"][0]
    top_confidence = result["scores"][0]
    return top_action, top_confidence
