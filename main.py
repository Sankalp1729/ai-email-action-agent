from email_reader import read_messages
from message_classifier import classify_message
from reward_model import calculate_reward
from visualizer import (
    plot_action_distribution,
    plot_reward_curve,
    plot_confidence_scores,
    plot_sender_response_pattern
)
from dashboard import display_dashboard
import json
from collections import defaultdict

# Step 1: Read messages from the sample JSON
messages = read_messages()

# Step 2: Initialize tracking structures
decision_log = []
action_distribution = defaultdict(int)
reward_over_time = []
confidence_scores = []
sender_response = {}

# Step 3: Loop through messages and take actions
for msg in messages:
    decision = classify_message(msg)
    reward = calculate_reward(msg, decision.action)

    # Log details
    decision_log.append({
        'sender': msg.sender,
        'subject': msg.subject,
        'content': msg.content,
        'action': decision.action,
        'confidence': decision.confidence,
        'reward': reward
    })

    action_distribution[decision.action] += 1
    reward_over_time.append(reward)
    confidence_scores.append(decision.confidence)

    # Track sender-based action pattern
    sender = msg.sender
    action = decision.action
    if sender not in sender_response:
        sender_response[sender] = {'Reply': 0, 'Later': 0, 'Forward': 0, 'Archive': 0}
    sender_response[sender][action] += 1

# Step 4: Save decision log to JSON
with open('decision_log.json', 'w') as f:
    json.dump(decision_log, f, indent=2)
print("✅ Decision log saved to 'decision_log.json'")

# Step 5: Plot graphs
plot_action_distribution(dict(action_distribution))
plot_reward_curve(reward_over_time)
plot_confidence_scores(confidence_scores)
plot_sender_response_pattern(sender_response)

# Step 6: Display dashboard
stats = {
    'action_distribution': dict(action_distribution),
    'reward_over_time': reward_over_time,
    'confidence_scores': confidence_scores,
    'sender_response': sender_response
}
display_dashboard(stats)
