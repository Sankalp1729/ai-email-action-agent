import json
from dataclasses import dataclass

@dataclass
class Message:
    sender: str
    subject: str
    content: str  # renamed from body to content

def read_messages(file_path='sample_emails.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    messages = [Message(**msg) for msg in data]
    return messages

