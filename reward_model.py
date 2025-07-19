# reward.py

def compute_reward(action, confidence, sentiment):
    """
    Reward logic:
    - Higher confidence gets more reward.
    - Positive sentiment is good for replies.
    - Urgent actions like 'Reply' have a higher base reward.
    """
    base_reward = 0.0

    # Ensure confidence is a float, not a tuple
    if isinstance(confidence, tuple):
        confidence = confidence[0]

    if action == "Reply":
        base_reward += 1.5
    elif action == "Forward":
        base_reward += 1.0
    elif action == "Later":
        base_reward += 0.5
    elif action == "Archive":
        base_reward += 0.1

    # Add confidence directly
    base_reward += confidence

    # Add bonus for positive sentiment
    if sentiment == "POSITIVE":
        base_reward += 0.2
    elif sentiment == "NEGATIVE":
        base_reward -= 0.2

    return round(base_reward, 2)
