import csv
from datetime import datetime
import os

def log_to_csv(data, filename="outputs/logs.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writerow([
                "Timestamp", "Sender", "Subject", "Content",
                "Action", "Confidence", "Sentiment", "Reward", "Reply"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["from"],
            data["subject"],
            data["body"],
            data["action"],
            round(data["confidence"], 2),
            data["sentiment"],
            data["reward"],
            data["reply"]
        ])
