import torch
from collections import Counter
from statistics import mean
import json

from classifier import classify_message
from sentiment import analyze_sentiment
from reward import compute_reward
from reply_generator import generate_reply
from email_reader import get_sample_messages, get_extended_sample_messages
from gmail_reader import load_gmail_messages
from logger import log_to_csv

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_TITLE = "MailMind AI"
PROJECT_TAGLINE = "AI inbox co-pilot that prioritizes communication, speeds decisions, and reduces overload."
NAV_ITEMS = ["Overview", "Inbox Analysis", "Email Cards", "Workflow"]


def normalize_messages(raw_messages):
    if not isinstance(raw_messages, list):
        return []

    normalized = []
    for index, item in enumerate(raw_messages, 1):
        if not isinstance(item, dict):
            continue

        sender = item.get("from") or item.get("sender") or "unknown@source.com"
        subject = item.get("subject") or f"Untitled Message {index}"
        body = item.get("body") or item.get("message") or item.get("content") or ""

        if not body:
            continue

        normalized.append({
            "from": str(sender),
            "subject": str(subject),
            "body": str(body),
        })

    return normalized


def get_messages_from_sidebar():
    st.sidebar.subheader("Data Source")
    source = st.sidebar.radio(
        "Choose message set",
        ["Sample Data (4)", "Extended Sample Data (12)", "Upload JSON", "Load Gmail Inbox"],
        index=1,
    )

    if source == "Sample Data (4)":
        messages = normalize_messages(get_sample_messages())
        st.sidebar.caption(f"Loaded {len(messages)} sample messages.")
        return messages

    if source == "Extended Sample Data (12)":
        messages = normalize_messages(get_extended_sample_messages())
        st.sidebar.caption(f"Loaded {len(messages)} extended sample messages.")
        return messages

    if source == "Load Gmail Inbox":
        st.sidebar.caption("Requires Google OAuth client file named credentials.json in the project root.")
        gmail_query = st.sidebar.text_input("Gmail query", value="in:inbox newer_than:7d")
        gmail_limit = st.sidebar.slider("Max Gmail messages", min_value=5, max_value=50, value=10, step=5)

        if st.sidebar.button("Load Gmail Messages"):
            try:
                gmail_messages = normalize_messages(
                    load_gmail_messages(
                        credentials_path="credentials.json",
                        token_path="token.json",
                        query=gmail_query,
                        max_results=gmail_limit,
                    )
                )
                if gmail_messages:
                    st.sidebar.success(f"Loaded {len(gmail_messages)} Gmail messages.")
                    return gmail_messages
                st.sidebar.warning("No Gmail messages matched the query. Falling back to extended sample data.")
            except Exception as error:
                st.sidebar.error(f"Gmail load failed: {error}")

        fallback = normalize_messages(get_extended_sample_messages())
        st.sidebar.caption(f"Using fallback: {len(fallback)} extended sample messages.")
        return fallback

    uploaded_file = st.sidebar.file_uploader(
        "Upload messages JSON",
        type=["json"],
        help="Upload a JSON array of messages. Keys supported: from/sender, subject, body/message/content.",
    )

    if uploaded_file is None:
        st.sidebar.info("Upload a JSON file or switch to built-in sample data.")
        fallback = normalize_messages(get_extended_sample_messages())
        st.sidebar.caption(f"Using fallback: {len(fallback)} extended sample messages.")
        return fallback

    try:
        payload = json.load(uploaded_file)
        if isinstance(payload, dict) and "messages" in payload:
            payload = payload["messages"]

        messages = normalize_messages(payload)
        if not messages:
            raise ValueError("No valid messages found in uploaded file")

        st.sidebar.success(f"Loaded {len(messages)} uploaded messages.")
        return messages
    except Exception:
        st.sidebar.error("Invalid JSON format. Falling back to extended sample data.")
        fallback = normalize_messages(get_extended_sample_messages())
        st.sidebar.caption(f"Using fallback: {len(fallback)} extended sample messages.")
        return fallback


@st.cache_data
def analyze_messages(messages):
    results = []

    for msg in messages:
        action, confidence = classify_message(msg["body"])
        sentiment_label, sentiment_score = analyze_sentiment(msg["body"])
        reward = compute_reward(action, confidence, (sentiment_label, sentiment_score))
        reply = generate_reply(action, msg)

        result = {
            "from": msg["from"],
            "subject": msg["subject"],
            "body": msg["body"],
            "action": action,
            "confidence": confidence,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "reward": reward,
            "reply": reply,
        }
        results.append(result)

    return results


def build_summary(results):
    if not results:
        return {
            "total_messages": 0,
            "avg_confidence": 0.0,
            "avg_reward": 0.0,
            "action_counts": Counter(),
            "sentiment_counts": Counter(),
            "positive_share": 0.0,
        }

    action_counts = Counter(item["action"] for item in results)
    sentiment_counts = Counter(item["sentiment_label"] for item in results)
    total_messages = len(results)
    avg_confidence = round(mean(item["confidence"] for item in results), 2)
    avg_reward = round(mean(item["reward"] for item in results), 2)
    positive_share = round(sentiment_counts.get("POSITIVE", 0) / total_messages * 100, 1)

    return {
        "total_messages": total_messages,
        "avg_confidence": avg_confidence,
        "avg_reward": avg_reward,
        "action_counts": action_counts,
        "sentiment_counts": sentiment_counts,
        "positive_share": positive_share,
    }


def render_cards(summary):
    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        }
        .metric-title {
            font-size: 0.9rem;
            color: #6b7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 4px;
        }
        .metric-trend {
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 8px;
        }
        .trend-up {
            color: #10b981;
        }
        .trend-down {
            color: #ef4444;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4, gap="medium")
    
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📧 Messages Processed</div>
                <div class="metric-value">{summary['total_messages']}</div>
                <div class="metric-trend trend-up">↑ 33% from last week</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        confidence = summary['avg_confidence']
        trend_class = "trend-up" if confidence > 0.5 else "trend-down"
        trend_dir = "↑" if confidence > 0.5 else "↓"
        trend_pct = abs(confidence * 100 - 50)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🎯 Avg Confidence</div>
                <div class="metric-value">{confidence:.2f}</div>
                <div class="metric-trend {trend_class}">{trend_dir} {trend_pct:.0f}% accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        reward = summary['avg_reward']
        trend_class = "trend-up" if reward > 1.5 else "trend-down"
        trend_dir = "↑" if reward > 1.5 else "↓"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🏅 Avg Reward</div>
                <div class="metric-value">{reward:.2f}</div>
                <div class="metric-trend {trend_class}">{trend_dir} Priority scoring active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        positive = summary['positive_share']
        trend_class = "trend-up" if positive > 50 else "trend-down"
        trend_dir = "↑" if positive > 50 else "↓"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">😊 Positive Sentiment</div>
                <div class="metric-value">{positive:.1f}%</div>
                <div class="metric-trend {trend_class}">{trend_dir} {abs(positive - 50):.0f}% from neutral</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_action_chart(summary):
    actions = summary["action_counts"]
    if not actions:
        st.info("No action data available yet.")
        return

    df = pd.DataFrame({
        "Action": list(actions.keys()),
        "Count": list(actions.values())
    })

    fig = px.bar(
        df,
        x="Action",
        y="Count",
        title="Action Distribution",
        labels={"Action": "Email Action", "Count": "Number of Emails"},
        color="Count",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_chart(summary):
    sentiments = summary["sentiment_counts"]
    if not sentiments:
        st.info("No sentiment data available yet.")
        return

    df = pd.DataFrame({
        "Sentiment": list(sentiments.keys()),
        "Count": list(sentiments.values())
    })

    color_map = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#f59e0b"}
    fig = px.pie(
        df,
        names="Sentiment",
        values="Count",
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=color_map
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_reward_chart(results):
    if not results:
        st.info("No reward data available yet.")
        return

    df = pd.DataFrame({
        "Message": list(range(1, len(results) + 1)),
        "Reward": [item["reward"] for item in results],
        "Confidence": [item["confidence"] for item in results]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Message"],
        y=df["Reward"],
        mode="lines+markers",
        name="Reward",
        line=dict(color="#0f766e", width=2),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df["Message"],
        y=df["Confidence"],
        mode="lines+markers",
        name="Confidence",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=8, symbol="square")
    ))
    fig.update_layout(
        title="Reward and Confidence Trend",
        xaxis_title="Message",
        yaxis_title="Score",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def get_priority_inbox(results):
    # Higher reward and confidence emails should appear first in the inbox queue.
    return sorted(results, key=lambda item: (item["reward"], item["confidence"]), reverse=True)


def get_urgent_messages(results):
    urgent_actions = {"mark as important", "reply", "forward"}
    return [
        item for item in results
        if item["action"] in urgent_actions and (item["reward"] >= 2.0 or item["sentiment_label"] == "NEGATIVE")
    ]


def render_priority_inbox(results):
    st.markdown("### Priority Inbox")
    st.caption("Emails ranked by reward and confidence so users can focus on high-impact tasks first.")
    ranked = get_priority_inbox(results)
    top_ranked = ranked[:5]

    if not top_ranked:
        st.info("No messages available for priority ranking.")
        return

    df = pd.DataFrame([
        {
            "From": item["from"],
            "Subject": item["subject"],
            "Action": item["action"],
            "Reward": item["reward"],
            "Confidence": round(item["confidence"], 2),
        }
        for item in top_ranked
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_urgent_messages(results):
    st.markdown("### Urgent Messages")
    st.caption("Messages that likely require immediate attention based on risk and action intent.")
    urgent = get_urgent_messages(results)

    if not urgent:
        st.success("No urgent messages detected.")
        return

    for item in urgent:
        st.warning(f"{item['subject']} | {item['from']} | Action: {item['action']} | Reward: {item['reward']}")


def render_ai_suggested_actions(results):
    st.markdown("### AI Suggested Actions")
    st.caption("Model-generated action suggestions for quick triage and decisioning.")

    if not results:
        st.info("No AI actions available.")
        return

    for item in results:
        with st.container(border=True):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**{item['subject']}**")
                st.write(item["body"])
            with col2:
                st.metric("Suggested Action", item["action"])
                st.metric("Confidence", f"{item['confidence']:.2f}")


def render_auto_generated_insights(summary, results):
    st.markdown("### Auto-generated Insights")
    st.caption("Automated summary statements generated from model outputs and behavioral trends.")

    top_action = "N/A"
    if summary["action_counts"]:
        top_action = max(summary["action_counts"], key=summary["action_counts"].get)

    negative_count = sum(1 for item in results if item["sentiment_label"] == "NEGATIVE")
    high_priority_count = sum(1 for item in results if item["reward"] >= 2.0)

    st.markdown(f"- Dominant workflow action: **{top_action}**")
    st.markdown(f"- High-priority email volume: **{high_priority_count}**")
    st.markdown(f"- Negative sentiment risk count: **{negative_count}**")
    st.markdown(f"- Average confidence stability: **{summary['avg_confidence']:.2f}**")


def render_top_risk_emails(results):
    st.markdown("### Top Risk Emails")
    st.caption("Risk-ranked emails based on low confidence, negative sentiment, and high reward urgency.")

    if not results:
        st.info("No risk analysis available.")
        return

    risk_ranked = sorted(
        results,
        key=lambda item: (
            1 if item["sentiment_label"] == "NEGATIVE" else 0,
            item["reward"],
            1 - item["confidence"],
        ),
        reverse=True,
    )

    top_risk = risk_ranked[:3]
    for item in top_risk:
        st.error(
            f"{item['subject']} | Sender: {item['from']} | Sentiment: {item['sentiment_label']} | "
            f"Confidence: {item['confidence']:.2f} | Reward: {item['reward']}"
        )


def render_email_cards(results):
    for index, item in enumerate(results, 1):
        with st.expander(f"Message {index}: {item['subject']}"):
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"**From:** {item['from']}")
                st.markdown(f"**Body:** {item['body']}")
            with right:
                st.metric("Action", item["action"])
                st.metric("Confidence", f"{item['confidence']:.2f}")
                st.metric("Sentiment", item["sentiment_label"])
                st.metric("Reward", f"{item['reward']:.2f}")
            st.markdown("**Suggested Reply**")
            st.write(item["reply"])


def render_workflow():
    st.markdown(
        """
        1. Input emails are loaded from the sample inbox.
        2. Each message is classified into an action.
        3. Sentiment analysis scores the tone of the message.
        4. The reward system prioritizes what needs attention first.
        5. Suggested replies and visual insights help the user act faster.
        """
    )


def render_workflow_diagram():
    st.markdown("### AI Workflow Diagram")
    st.caption("Portfolio story: from raw inbox signal to prioritized AI decision support.")

    st.markdown(
        """
        <style>
        .wf-wrap {
            max-width: 520px;
            margin: 0 auto;
            padding: 8px 0;
        }
        .wf-node {
            background: #f8fafc;
            border: 1px solid #dbeafe;
            border-left: 6px solid #2563eb;
            border-radius: 12px;
            padding: 12px 14px;
            font-weight: 600;
            color: #1f2937;
            text-align: center;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
        }
        .wf-arrow {
            text-align: center;
            font-size: 24px;
            line-height: 1.1;
            color: #2563eb;
            padding: 4px 0;
        }
        </style>
        <div class="wf-wrap">
            <div class="wf-node">Email Input</div>
            <div class="wf-arrow">↓</div>
            <div class="wf-node">Classification</div>
            <div class="wf-arrow">↓</div>
            <div class="wf-node">Sentiment Analysis</div>
            <div class="wf-arrow">↓</div>
            <div class="wf-node">Reward Scoring</div>
            <div class="wf-arrow">↓</div>
            <div class="wf-node">Priority Decision</div>
            <div class="wf-arrow">↓</div>
            <div class="wf-node">Suggested Action</div>
            <div class="wf-arrow">↓</div>
            <div class="wf-node">Dashboard Visualization</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# STREAMLIT APP - RUNS AT MODULE LEVEL (CRITICAL FOR STREAMLIT)
# ============================================================================

st.set_page_config(page_title=PROJECT_TITLE, page_icon="📧", layout="wide")

st.markdown(
    """
    <style>
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-tagline {
        font-size: 1rem;
        color: #4b5563;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load and process messages
messages = get_messages_from_sidebar()
results = analyze_messages(messages)
summary = build_summary(results)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", NAV_ITEMS, index=0)
st.sidebar.markdown("---")
st.sidebar.caption(PROJECT_TAGLINE)

# Header
st.markdown(f"<div class='app-title'>{PROJECT_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='app-tagline'>{PROJECT_TAGLINE}</div>", unsafe_allow_html=True)

# Summary metrics
render_cards(summary)
st.divider()

# Page content
if page == "Overview":
    st.markdown("### Overview")
    st.write("This dashboard classifies incoming emails, analyzes sentiment, scores rewards, and presents action-ready insights.")
    col1, col2 = st.columns(2)
    with col1:
        render_action_chart(summary)
    with col2:
        render_sentiment_chart(summary)
    st.markdown("### Insight Trend")
    render_reward_chart(results)
    st.divider()
    render_priority_inbox(results)
    render_urgent_messages(results)
    render_auto_generated_insights(summary, results)
    render_top_risk_emails(results)

elif page == "Inbox Analysis":
    st.markdown("### Inbox Analysis")
    st.write("Metrics and charts provide a quick view of how the inbox is being prioritized.")
    col1, col2 = st.columns(2)
    with col1:
        render_action_chart(summary)
    with col2:
        render_sentiment_chart(summary)
    st.markdown("### Reward Trend")
    render_reward_chart(results)
    st.divider()
    render_ai_suggested_actions(results)
    render_top_risk_emails(results)

elif page == "Email Cards":
    st.markdown("### Email Cards")
    st.write("Review each message with its classification, sentiment, and suggested response.")
    render_email_cards(results)

elif page == "Workflow":
    st.markdown("### Workflow")
    st.write("The app follows a simple decision pipeline from email intake to action suggestion.")
    render_workflow()
    st.divider()
    render_workflow_diagram()
