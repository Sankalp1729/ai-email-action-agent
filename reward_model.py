def calculate_reward(message, action):
    if "urgent" in message.subject.lower() and action == 'Reply':
        return 1.0
    elif action == 'Archive':
        return 0.2
    elif action == 'Forward':
        return 0.5
    elif action == 'Later':
        return 0.3
    else:
        return 0.0
