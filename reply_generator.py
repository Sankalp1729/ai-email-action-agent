def generate_reply(action, message):
    """
    Generate a reply based on the action and message content.
    
    Args:
        action (str): The predicted action.
        message (dict): The message dictionary containing 'subject' and 'body'.
    
    Returns:
        str: Generated reply text.
    """
    body = message.get("body", "")
    if isinstance(body, dict):
        # If body is mistakenly a dict, extract 'content' key if it exists
        body = body.get("content", "")

    subject = message.get("subject", "your email")

    if action == "reply":
        if "project" in body.lower():
            return f"Hi,\n\nThank you for the update on the project. I will review and get back to you shortly.\n\nBest regards,"
        elif "meeting" in body.lower():
            return f"Hi,\n\nThanks for scheduling the meeting. Looking forward to it.\n\nBest regards,"
        else:
            return f"Hi,\n\nThanks for your email regarding {subject.lower()}. I'll get back to you soon.\n\nBest regards,"
    elif action == "forward":
        return f"Forwarding this email regarding '{subject}' as it may require your attention."
    elif action == "mark as important":
        return f"Marked as important: '{subject}'"
    elif action == "delete":
        return f"This message regarding '{subject}' has been deleted."
    elif action == "ignore":
        return f"No action taken for the email regarding '{subject}'."
    else:
        return "No reply generated."
