import matplotlib.pyplot as plt
import numpy as np

def plot_action_distribution(action_distribution):
    actions = list(action_distribution.keys())
    counts = list(action_distribution.values())
    plt.figure(figsize=(6, 4))
    plt.bar(actions, counts, color='skyblue')
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_reward_curve(rewards):
    plt.figure(figsize=(6, 4))
    plt.plot(rewards, marker='o', linestyle='--', color='green')
    plt.title("Reward Over Time")
    plt.xlabel("Message #")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()

def plot_confidence_scores(confidences):
    plt.figure(figsize=(6, 4))
    plt.plot(confidences, marker='x', linestyle='-', color='orange')
    plt.title("Confidence Scores")
    plt.xlabel("Message #")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.show()

def plot_sender_response_pattern(sender_response):
    senders = list(sender_response.keys())
    actions = ['Reply', 'Later', 'Forward', 'Archive']
    bar_width = 0.2
    index = np.arange(len(senders))
    colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0']

    plt.figure(figsize=(10, 6))
    for i, action in enumerate(actions):
        counts = [sender_response[sender][action] for sender in senders]
        plt.bar(index + i * bar_width, counts, bar_width, label=action, color=colors[i])

    plt.xlabel('Sender')
    plt.ylabel('Count')
    plt.title('Sender Response Pattern')
    plt.xticks(index + bar_width * 1.5, senders, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
