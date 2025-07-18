import random

class Decision:
    def __init__(self, action, confidence):
        self.action = action
        self.confidence = confidence

def classify_message(message):
    actions = ['Reply', 'Later', 'Forward', 'Archive']
    action = random.choice(actions)
    confidence = round(random.uniform(0.6, 1.0), 2)
    return Decision(action, confidence)
