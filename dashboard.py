def display_dashboard(stats):
    print("\n📊 Dashboard Summary 📊")
    print("-" * 40)
    print("✅ Action Distribution:")
    for action, count in stats['action_distribution'].items():
        print(f"  {action}: {count}")

    print("\n📈 Avg Reward:", round(sum(stats['reward_over_time']) / len(stats['reward_over_time']), 2))
    print("🎯 Avg Confidence:", round(sum(stats['confidence_scores']) / len(stats['confidence_scores']), 2))

    print("\n📨 Sender Response Pattern:")
    for sender, actions in stats['sender_response'].items():
        print(f"  {sender}: {actions}")
