import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import re
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Message data structure"""
    sender: str
    subject: str
    priority: str
    timestamp: str
    body: Optional[str] = None
    
@dataclass
class ActionDecision:
    """Action decision with metadata"""
    action: str
    confidence: float
    reason: str
    timestamp: str
    reward: float = 0.0

class EmailClassifier:
    """Rule-based email classifier with sentiment analysis"""
    
    def __init__(self):
        self.urgent_keywords = ['urgent', 'asap', 'immediate', 'emergency', 'deadline', 'critical']
        self.spam_keywords = ['offer', 'discount', 'free', 'winner', 'congratulations', 'click here']
        self.work_domains = ['company.com', 'work.com', 'corp.com', 'office.com']
        self.newsletter_domains = ['newsletter.com', 'news.com', 'blog.com', 'updates.com']
        
    def extract_features(self, message: Message) -> Dict:
        """Extract features from message for classification"""
        features = {
            'is_high_priority': message.priority.lower() == 'high',
            'is_work_email': any(domain in message.sender for domain in self.work_domains),
            'is_newsletter': any(domain in message.sender for domain in self.newsletter_domains),
            'has_urgent_words': any(word in message.subject.lower() for word in self.urgent_keywords),
            'has_spam_words': any(word in message.subject.lower() for word in self.spam_keywords),
            'sender_frequency': self._get_sender_frequency(message.sender),
            'time_of_day': self._get_time_category(message.timestamp),
            'subject_length': len(message.subject),
            'sentiment_score': self._analyze_sentiment(message.subject)
        }
        return features
    
    def _get_sender_frequency(self, sender: str) -> str:
        """Simulate sender frequency analysis"""
        # In real implementation, this would check historical data
        if any(domain in sender for domain in self.work_domains):
            return 'frequent'
        elif any(domain in sender for domain in self.newsletter_domains):
            return 'regular'
        else:
            return 'rare'
    
    def _get_time_category(self, timestamp: str) -> str:
        """Categorize time of day"""
        try:
            time_obj = datetime.strptime(timestamp, "%Y-%m-%d %I:%M %p")
            hour = time_obj.hour
            if 9 <= hour <= 17:
                return 'business_hours'
            elif 6 <= hour <= 9 or 17 <= hour <= 21:
                return 'extended_hours'
            else:
                return 'off_hours'
        except:
            return 'unknown'
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ['thanks', 'great', 'excellent', 'good', 'happy', 'pleased']
        negative_words = ['urgent', 'problem', 'issue', 'error', 'failed', 'angry']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)

class RLAgent:
    """Reinforcement Learning Agent using Q-learning"""
    
    def __init__(self, actions=['reply', 'later', 'archive', 'forward'], 
                 learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.action_counts = defaultdict(int)
        self.total_reward = 0.0
        self.episode_rewards = []
        
    def get_state_key(self, features: Dict) -> str:
        """Convert features to state key"""
        key_features = [
            f"priority_{features['is_high_priority']}",
            f"work_{features['is_work_email']}",
            f"urgent_{features['has_urgent_words']}",
            f"spam_{features['has_spam_words']}",
            f"time_{features['time_of_day']}",
            f"freq_{features['sender_frequency']}"
        ]
        return "_".join(key_features)
    
    def choose_action(self, state: str) -> Tuple[str, float]:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: choose random action
            action = random.choice(self.actions)
            confidence = 0.3  # Low confidence for random actions
        else:
            # Exploit: choose best action
            q_values = self.q_table[state]
            if not q_values:
                action = random.choice(self.actions)
                confidence = 0.5
            else:
                action = max(q_values, key=q_values.get)
                max_q = max(q_values.values())
                min_q = min(q_values.values())
                confidence = 0.5 + 0.5 * (max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-6)
        
        return action, min(confidence, 1.0)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str = None):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]
        
        if next_state:
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0
        else:
            max_next_q = 0
            
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        self.action_counts[action] += 1
        self.total_reward += reward
        
    def decay_epsilon(self):
        """Decay exploration rate over time"""
        self.epsilon = max(0.01, self.epsilon * 0.995)

class RewardFunction:
    """Reward function for email classification"""
    
    def calculate_reward(self, action: str, features: Dict, message: Message) -> float:
        """Calculate reward based on action and message features"""
        reward = 0.0
        
        # High priority messages should be replied to
        if features['is_high_priority'] and action == 'reply':
            reward += 2.0
        elif features['is_high_priority'] and action == 'archive':
            reward -= 2.0
            
        # Urgent messages should be replied to
        if features['has_urgent_words'] and action == 'reply':
            reward += 1.5
        elif features['has_urgent_words'] and action == 'archive':
            reward -= 1.5
            
        # Spam should be archived
        if features['has_spam_words'] and action == 'archive':
            reward += 1.0
        elif features['has_spam_words'] and action == 'reply':
            reward -= 2.0
            
        # Work emails during business hours should be prioritized
        if features['is_work_email'] and features['time_of_day'] == 'business_hours':
            if action == 'reply':
                reward += 1.0
            elif action == 'archive':
                reward -= 0.5
                
        # Newsletters should be archived or scheduled for later
        if features['is_newsletter']:
            if action in ['archive', 'later']:
                reward += 0.5
            elif action == 'reply':
                reward -= 1.0
                
        # Unknown senders might need forwarding
        if features['sender_frequency'] == 'rare' and action == 'forward':
            reward += 1.0
            
        return reward

class ActionAgent:
    """Main AI Action Agent"""
    
    def __init__(self):
        self.classifier = EmailClassifier()
        self.rl_agent = RLAgent()
        self.reward_function = RewardFunction()
        self.decision_log = []
        self.metrics = {
            'total_processed': 0,
            'actions_taken': defaultdict(int),
            'confidence_scores': [],
            'rewards_over_time': [],
            'sender_patterns': defaultdict(lambda: defaultdict(int))
        }
        
    def process_message(self, message: Message, feedback: Optional[str] = None) -> ActionDecision:
        """Process a single message and return action decision"""
        # Extract features
        features = self.classifier.extract_features(message)
        state = self.rl_agent.get_state_key(features)
        
        # Choose action
        action, confidence = self.rl_agent.choose_action(state)
        
        # Calculate reward
        reward = self.reward_function.calculate_reward(action, features, message)
        
        # Apply feedback if provided
        if feedback:
            feedback_reward = self._process_feedback(feedback, action)
            reward += feedback_reward
            
        # Update Q-table
        self.rl_agent.update_q_value(state, action, reward)
        
        # Generate decision reason
        reason = self._generate_reason(action, features, confidence)
        
        # Create decision object
        decision = ActionDecision(
            action=action,
            confidence=confidence,
            reason=reason,
            timestamp=datetime.now().isoformat(),
            reward=reward
        )
        
        # Log decision
        self.decision_log.append({
            'message': asdict(message),
            'decision': asdict(decision),
            'features': features,
            'state': state
        })
        
        # Update metrics
        self._update_metrics(message, decision)
        
        # Decay exploration
        self.rl_agent.decay_epsilon()
        
        return decision
    
    def _process_feedback(self, feedback: str, action: str) -> float:
        """Process user feedback and return additional reward"""
        feedback_lower = feedback.lower()
        if 'correct' in feedback_lower or 'good' in feedback_lower:
            return 1.0
        elif 'wrong' in feedback_lower or 'bad' in feedback_lower:
            return -1.0
        else:
            return 0.0
    
    def _generate_reason(self, action: str, features: Dict, confidence: float) -> str:
        """Generate human-readable reason for decision"""
        reasons = []
        
        if features['is_high_priority']:
            reasons.append("High priority message")
        if features['has_urgent_words']:
            reasons.append("Contains urgent keywords")
        if features['is_work_email']:
            reasons.append("From work domain")
        if features['has_spam_words']:
            reasons.append("Contains spam indicators")
        if features['time_of_day'] == 'business_hours':
            reasons.append("Received during business hours")
            
        base_reason = f"Action: {action.upper()} - "
        if reasons:
            base_reason += "; ".join(reasons)
        else:
            base_reason += "Based on general patterns"
            
        base_reason += f" (Confidence: {confidence:.2f})"
        return base_reason
    
    def _update_metrics(self, message: Message, decision: ActionDecision):
        """Update system metrics"""
        self.metrics['total_processed'] += 1
        self.metrics['actions_taken'][decision.action] += 1
        self.metrics['confidence_scores'].append(decision.confidence)
        self.metrics['rewards_over_time'].append(decision.reward)
        self.metrics['sender_patterns'][message.sender][decision.action] += 1
    
    def batch_process(self, messages: List[Message], feedback_data: Optional[Dict] = None) -> List[ActionDecision]:
        """Process multiple messages"""
        decisions = []
        
        for i, message in enumerate(messages):
            feedback = feedback_data.get(i) if feedback_data else None
            decision = self.process_message(message, feedback)
            decisions.append(decision)
            
        return decisions
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        if not self.decision_log:
            return {"error": "No decisions logged yet"}
            
        avg_confidence = np.mean(self.metrics['confidence_scores'])
        avg_reward = np.mean(self.metrics['rewards_over_time'])
        
        return {
            'total_processed': self.metrics['total_processed'],
            'average_confidence': avg_confidence,
            'average_reward': avg_reward,
            'action_distribution': dict(self.metrics['actions_taken']),
            'exploration_rate': self.rl_agent.epsilon,
            'total_states_learned': len(self.rl_agent.q_table)
        }

class Visualizer:
    """Visualization component for metrics and analysis"""
    
    def __init__(self, agent: ActionAgent):
        self.agent = agent
        plt.style.use('seaborn-v0_8')
        
    def plot_action_distribution(self, figsize=(10, 6)):
        """Plot distribution of actions taken"""
        actions = list(self.agent.metrics['actions_taken'].keys())
        counts = list(self.agent.metrics['actions_taken'].values())
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.bar(actions, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=actions, autopct='%1.1f%%', 
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        plt.title('Action Distribution (%)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confidence_trends(self, figsize=(12, 4)):
        """Plot confidence scores over time"""
        confidence_scores = self.agent.metrics['confidence_scores']
        if not confidence_scores:
            print("No confidence data available")
            return
            
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.plot(confidence_scores, alpha=0.7, color='#45B7D1')
        plt.title('Confidence Scores Over Time')
        plt.xlabel('Message Number')
        plt.ylabel('Confidence Score')
        
        # Add moving average
        window = min(10, len(confidence_scores) // 3)
        if window > 1:
            moving_avg = pd.Series(confidence_scores).rolling(window=window).mean()
            plt.plot(moving_avg, color='#FF6B6B', linewidth=2, label=f'Moving Average ({window})')
            plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(confidence_scores, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def plot_reward_trends(self, figsize=(12, 4)):
        """Plot reward trends and learning progress"""
        rewards = self.agent.metrics['rewards_over_time']
        if not rewards:
            print("No reward data available")
            return
            
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.7, color='#FFA07A')
        plt.title('Rewards Over Time')
        plt.xlabel('Message Number')
        plt.ylabel('Reward')
        
        # Add moving average
        window = min(10, len(rewards) // 3)
        if window > 1:
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            plt.plot(moving_avg, color='#FF6B6B', linewidth=2, label=f'Moving Average ({window})')
            plt.legend()
        
        plt.subplot(1, 2, 2)
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, color='#45B7D1', linewidth=2)
        plt.title('Cumulative Rewards')
        plt.xlabel('Message Number')
        plt.ylabel('Cumulative Reward')
        
        plt.tight_layout()
        plt.show()
    
    def plot_sender_patterns(self, figsize=(12, 8)):
        """Plot sender-response patterns"""
        sender_patterns = self.agent.metrics['sender_patterns']
        if not sender_patterns:
            print("No sender pattern data available")
            return
            
        # Create heatmap data
        senders = list(sender_patterns.keys())
        actions = ['reply', 'later', 'archive', 'forward']
        
        heatmap_data = []
        for sender in senders:
            row = [sender_patterns[sender][action] for action in actions]
            heatmap_data.append(row)
            
        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data, 
                   xticklabels=actions, 
                   yticklabels=senders, 
                   annot=True, 
                   fmt='d',
                   cmap='YlOrRd')
        plt.title('Sender-Response Patterns')
        plt.xlabel('Action')
        plt.ylabel('Sender')
        plt.tight_layout()
        plt.show()
    
    def generate_dashboard(self, figsize=(16, 12)):
        """Generate comprehensive dashboard"""
        plt.figure(figsize=figsize)
        
        # Action distribution
        plt.subplot(2, 3, 1)
        actions = list(self.agent.metrics['actions_taken'].keys())
        counts = list(self.agent.metrics['actions_taken'].values())
        plt.bar(actions, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        plt.title('Action Distribution')
        plt.xticks(rotation=45)
        
        # Confidence trends
        plt.subplot(2, 3, 2)
        confidence_scores = self.agent.metrics['confidence_scores']
        if confidence_scores:
            plt.plot(confidence_scores, alpha=0.7, color='#45B7D1')
            plt.title('Confidence Over Time')
            plt.xlabel('Message #')
            plt.ylabel('Confidence')
        
        # Reward trends
        plt.subplot(2, 3, 3)
        rewards = self.agent.metrics['rewards_over_time']
        if rewards:
            plt.plot(rewards, alpha=0.7, color='#FFA07A')
            plt.title('Rewards Over Time')
            plt.xlabel('Message #')
            plt.ylabel('Reward')
        
        # Cumulative rewards
        plt.subplot(2, 3, 4)
        if rewards:
            cumulative_rewards = np.cumsum(rewards)
            plt.plot(cumulative_rewards, color='#4ECDC4', linewidth=2)
            plt.title('Cumulative Learning')
            plt.xlabel('Message #')
            plt.ylabel('Cumulative Reward')
        
        # Confidence distribution
        plt.subplot(2, 3, 5)
        if confidence_scores:
            plt.hist(confidence_scores, bins=15, alpha=0.7, color='#45B7D1')
            plt.title('Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
        
        # Performance metrics
        plt.subplot(2, 3, 6)
        report = self.agent.get_performance_report()
        metrics_text = f"""
        Total Processed: {report.get('total_processed', 0)}
        Avg Confidence: {report.get('average_confidence', 0):.3f}
        Avg Reward: {report.get('average_reward', 0):.3f}
        States Learned: {report.get('total_states_learned', 0)}
        Exploration Rate: {report.get('exploration_rate', 0):.3f}
        """
        plt.text(0.1, 0.5, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.title('Performance Metrics')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Sample data generator for testing
def generate_sample_data(num_messages: int = 50) -> List[Message]:
    """Generate sample email data for testing"""
    
    senders = [
        'hr@company.com', 'boss@company.com', 'team@company.com',
        'newsletter@techblog.com', 'updates@news.com', 'offers@shopping.com',
        'support@service.com', 'noreply@social.com', 'alerts@bank.com',
        'friend@gmail.com', 'family@yahoo.com', 'colleague@work.com'
    ]
    
    subjects = [
        'Interview Schedule', 'Project Update', 'Meeting Tomorrow',
        'Urgent: Server Down', 'Weekly Newsletter', 'Special Offer',
        'Account Alert', 'Happy Birthday', 'Conference Invitation',
        'Deadline Reminder', 'Thank You', 'System Maintenance'
    ]
    
    priorities = ['High', 'Medium', 'Low']
    
    messages = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(num_messages):
        timestamp = base_time + timedelta(hours=random.randint(0, 168))
        
        message = Message(
            sender=random.choice(senders),
            subject=random.choice(subjects),
            priority=random.choice(priorities),
            timestamp=timestamp.strftime("%Y-%m-%d %I:%M %p")
        )
        messages.append(message)
    
    return messages

# Example usage and testing
if __name__ == "__main__":
    # Initialize the AI Action Agent
    agent = ActionAgent()
    visualizer = Visualizer(agent)
    
    # Generate sample data
    print("Generating sample email data...")
    sample_messages = generate_sample_data(100)
    
    # Process messages
    print("Processing messages with AI Agent...")
    decisions = agent.batch_process(sample_messages)
    
    # Display some results
    print("\n" + "="*50)
    print("SAMPLE DECISIONS")
    print("="*50)
    
    for i, decision in enumerate(decisions[:5]):
        msg = sample_messages[i]
        print(f"\nMessage {i+1}:")
        print(f"From: {msg.sender}")
        print(f"Subject: {msg.subject}")
        print(f"Priority: {msg.priority}")
        print(f"Decision: {decision.action.upper()}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Reason: {decision.reason}")
        print(f"Reward: {decision.reward:.2f}")
        print("-" * 40)
    
    # Generate performance report
    print("\n" + "="*50)
    print("PERFORMANCE REPORT")
    print("="*50)
    
    report = agent.get_performance_report()
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Individual plots
    visualizer.plot_action_distribution()
    visualizer.plot_confidence_trends()
    visualizer.plot_reward_trends()
    visualizer.plot_sender_patterns()
    
    # Comprehensive dashboard
    visualizer.generate_dashboard()
    
    # Save decision log
    print("\nSaving decision log...")
    with open('decision_log.json', 'w') as f:
        json.dump(agent.decision_log, f, indent=2)
    
    print("AI Action Agent test completed successfully!")
    print(f"Processed {len(decisions)} messages")
    print(f"Decision log saved to 'decision_log.json'")

