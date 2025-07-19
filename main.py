import torch
from classifier import classify_message
from sentiment import analyze_sentiment
from reward import compute_reward
from reply_generator import generate_reply
from email_reader import get_sample_messages
from logger import log_to_csv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")
print("🔍 AI Email Action Agent Started...\n")

def main():
    messages = get_sample_messages()

    for idx, msg in enumerate(messages, 1):
        print(f"\n📨 Message {idx}")
        print(f"From: {msg['from']}")
        print(f"Subject: {msg['subject']}")
        print(f"Content: {msg['body']}")

        action, confidence = classify_message(msg["body"])
        sentiment = analyze_sentiment(msg["body"])
        reward = compute_reward(action, confidence, sentiment)
        reply = generate_reply(action, msg)

        print(f"🔧 Action: {action}")
        print(f"📊 Confidence: {round(confidence, 2)}")
        print(f"❤️ Sentiment: {sentiment}")
        print(f"🏅 Reward: {reward}")
        print(f"✉️ Reply:\n{reply}")

        # Log to CSV
        log_to_csv({
            "from": msg["from"],
            "subject": msg["subject"],
            "body": msg["body"],
            "action": action,
            "confidence": confidence,
            "sentiment": sentiment,
            "reward": reward,
            "reply": reply
        })

if __name__ == "__main__":
    main()
