def compute_reward(action, confidence_score, sentiment_score):
    """
    Compute a reward score based on action type, confidence, and sentiment.
    
    Args:
        action (str): The action predicted (e.g., reply, ignore).
        confidence_score (float): The confidence score of the prediction.
        sentiment_score (tuple): A tuple like ('POSITIVE', score)
    
    Returns:
        float: Computed reward.
    """
    reward = 0.0

    # Add confidence
    reward += confidence_score

    # Sentiment score is a tuple like ('POSITIVE', 0.93)
    sentiment_label, score = sentiment_score
    if sentiment_label == 'POSITIVE':
        reward += score
    elif sentiment_label == 'NEGATIVE':
        reward -= score
    else:
        reward += 0.0  # Neutral has no effect

    # Adjust reward based on action
    if action == "reply":
        reward += 1.0
    elif action == "ignore":
        reward -= 0.5
    elif action == "delete":
        reward -= 1.0
    elif action == "mark as important":
        reward += 1.5
    elif action == "forward":
        reward += 0.8

    return round(reward, 2)
