"""
📊 FEEDBACK ANALYTICS DASHBOARD
================================
Visualize and analyze user feedback data.

Run with: streamlit run feedback_dashboard.py
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from feedback_collector import FeedbackCollector

# Page config
st.set_page_config(
    page_title="Feedback Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Feedback Analytics Dashboard")
st.markdown("Analyze user feedback to improve the placement assistant")

# Initialize collector
collector = FeedbackCollector()

# Load data
try:
    feedback_data = collector.get_all_feedback()
    stats = collector.get_stats()
except Exception as e:
    st.error(f"Error loading feedback: {e}")
    st.stop()

if not feedback_data:
    st.info("No feedback collected yet. Use the main app and provide feedback!")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(feedback_data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# ========== METRICS ==========
st.header("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Feedback", stats['total'])

with col2:
    helpful_rate = stats['helpful_rate'] * 100
    st.metric("Helpful Rate", f"{helpful_rate:.1f}%")

with col3:
    avg_rating = stats['avg_rating']
    st.metric("Avg Rating", f"{avg_rating:.1f}/5.0")

with col4:
    agentic_count = df[df['metadata'].apply(lambda x: x.get('mode') == 'agentic' if isinstance(x, dict) else False)].shape[0]
    agentic_rate = (agentic_count / len(df)) * 100 if len(df) > 0 else 0
    st.metric("Agentic Usage", f"{agentic_rate:.1f}%")

st.markdown("---")

# ========== FEEDBACK OVER TIME ==========
st.header("📅 Feedback Over Time")

# Daily feedback counts
daily_counts = df.groupby('date').size().reset_index(name='count')
st.line_chart(daily_counts.set_index('date'))

st.markdown("---")

# ========== ISSUE ANALYSIS ==========
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Common Issues")
    
    if stats['common_issues']:
        issue_df = pd.DataFrame([
            {"Issue": issue, "Count": count}
            for issue, count in stats['common_issues'].items()
        ]).sort_values('Count', ascending=False)
        
        st.bar_chart(issue_df.set_index('Issue'))
    else:
        st.info("No issues reported yet")

with col2:
    st.subheader("⭐ Rating Distribution")
    
    if 'rating' in df.columns:
        rating_counts = df['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)
    else:
        st.info("No ratings yet")

st.markdown("---")

# ========== MODE COMPARISON ==========
st.header("🤖 Mode Performance Comparison")

# Extract mode from metadata
df['mode'] = df['metadata'].apply(lambda x: x.get('mode', 'unknown') if isinstance(x, dict) else 'unknown')

mode_stats = df.groupby('mode').agg({
    'helpful': lambda x: (x == True).sum() / len(x) * 100 if len(x) > 0 else 0,
    'rating': 'mean'
}).round(2)

if not mode_stats.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Helpful Rate by Mode")
        st.bar_chart(mode_stats['helpful'])
    
    with col2:
        st.subheader("Avg Rating by Mode")
        st.bar_chart(mode_stats['rating'])
else:
    st.info("Not enough data for mode comparison")

st.markdown("---")

# ========== RECENT SUGGESTIONS ==========
st.header("💬 Recent Suggestions")

recent_suggestions = df[df['suggestion'].str.len() > 0].sort_values('timestamp', ascending=False).head(10)

if not recent_suggestions.empty:
    for idx, row in recent_suggestions.iterrows():
        with st.expander(f"💭 {row['timestamp'].strftime('%Y-%m-%d %H:%M')} - Rating: {row.get('rating', 'N/A')}/5"):
            st.write(f"**Query:** {row['query'][:100]}...")
            st.write(f"**Suggestion:** {row['suggestion']}")
            st.write(f"**Issues:** {', '.join(row.get('issues', [])) if row.get('issues') else 'None'}")
            st.write(f"**Mode:** {row.get('mode', 'N/A')}")
else:
    st.info("No suggestions yet")

st.markdown("---")

# ========== QUERY TYPE PERFORMANCE ==========
st.header("🎯 Query Type Performance")

# Extract query_type from metadata
df['query_type'] = df['metadata'].apply(
    lambda x: x.get('query_type', 'N/A') if isinstance(x, dict) else 'N/A'
)

query_type_stats = df[df['query_type'] != 'N/A'].groupby('query_type').agg({
    'helpful': lambda x: (x == True).sum() / len(x) * 100 if len(x) > 0 else 0,
    'rating': 'mean',
    'query_type': 'count'
}).round(2)

query_type_stats.columns = ['Helpful Rate (%)', 'Avg Rating', 'Count']

if not query_type_stats.empty:
    st.dataframe(query_type_stats.sort_values('Count', ascending=False))
else:
    st.info("No query type data available")

st.markdown("---")

# ========== DETAILED FEEDBACK TABLE ==========
st.header("📋 Detailed Feedback")

# Prepare display dataframe
display_df = df[[
    'timestamp', 'query', 'helpful', 'rating', 'mode'
]].copy()

display_df['query'] = display_df['query'].str[:50] + '...'
display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

st.dataframe(display_df, use_container_width=True)

st.markdown("---")

# ========== INSIGHTS & RECOMMENDATIONS ==========
st.header("💡 Insights & Recommendations")

# Calculate insights
insights = []

if stats['helpful_rate'] < 0.7:
    insights.append("⚠️ **Low helpful rate (<70%):** Consider improving answer quality or adding more context")

if stats['avg_rating'] < 3.5:
    insights.append("⚠️ **Low average rating (<3.5):** Review common issues and improve problematic areas")

if 'incomplete' in stats['common_issues'] and stats['common_issues']['incomplete'] > 3:
    insights.append("📝 **Frequent incomplete answers:** Add more comprehensive information to source documents")

if 'wrong_info' in stats['common_issues'] and stats['common_issues']['wrong_info'] > 2:
    insights.append("❌ **Wrong information detected:** Verify and update source documents")

if 'slow' in stats['common_issues'] and stats['common_issues']['slow'] > 2:
    insights.append("⏱️ **Slow responses:** Consider optimizing retrieval or using faster models")

# Mode-specific insights
if 'basic' in mode_stats.index and 'agentic' in mode_stats.index:
    if mode_stats.loc['agentic', 'helpful'] > mode_stats.loc['basic', 'helpful'] + 10:
        insights.append("✅ **Agentic RAG performing better:** Consider defaulting to agentic mode for more queries")
    elif mode_stats.loc['basic', 'helpful'] > mode_stats.loc['agentic', 'helpful'] + 10:
        insights.append("✅ **Basic RAG performing better:** Agentic mode may be over-engineering simple queries")

if insights:
    for insight in insights:
        st.write(insight)
else:
    st.success("✅ No major issues detected! Keep up the good work!")

st.markdown("---")

# ========== ACTIONS ==========
st.header("🔧 Actions")

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear Old Feedback (>30 days)", use_container_width=True):
        removed = collector.clear_old_feedback(days=30)
        st.success(f"Removed {removed} old feedback entries")
        st.rerun()

with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Download feedback
st.markdown("---")
st.subheader("💾 Export Data")

if st.button("Download Feedback as JSON", use_container_width=True):
    feedback_json = json.dumps(feedback_data, indent=2)
    st.download_button(
        label="Download feedback.json",
        data=feedback_json,
        file_name=f"feedback_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )
