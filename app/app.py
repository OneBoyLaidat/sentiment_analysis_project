import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Phlex Sentiment Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 1. Loading Icon & Data Processing
with st.spinner('Accessing Phlex Data & Processing Sentiments...'):
    # Load Data
    df = pd.read_csv('data/complaints.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sentiment Logic
    def get_sentiment_details(text):
        analysis = TextBlob(text)
        score = analysis.sentiment.polarity
        if score > 0: return 'Positive', score
        elif score < 0: return 'Negative', score
        else: return 'Neutral', score

    # Apply analysis
    df[['sentiment', 'polarity']] = df['ticket_text'].apply(lambda x: pd.Series(get_sentiment_details(x)))

# 2. Header Section
st.title("📊 Phlex Payment: Customer Sentiment Analysis System")
st.markdown(f"""
        **Objective:** Prioritize high-urgency tickets and improve payment failure user experience.
""")

# 3. Sidebar Filters
st.sidebar.header("Dashboard Filters")
category = st.sidebar.multiselect("Filter by Category", options=df['category'].unique(), default=df['category'].unique())
date_range = st.sidebar.date_input("Select Date Range", 
                                   value=(df['timestamp'].min(), df['timestamp'].max()),
                                   min_value=df['timestamp'].min(), 
                                   max_value=df['timestamp'].max())

# Filter Logic
mask = (df['category'].isin(category)) & (df['timestamp'].dt.date.between(date_range[0], date_range[1]))
filtered_df = df.loc[mask]

# 4. Measure Views (KPIs)
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_tickets = len(filtered_df)
neg_count = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
avg_polarity = filtered_df['polarity'].mean()

col1.metric("Total Tickets", total_tickets)
col2.metric("Negative Ratio", f"{(neg_count/total_tickets)*100:.1f}%", delta="-2%" if neg_count < 500 else "+5%", delta_color="inverse")
col3.metric("Avg. Polarity", f"{avg_polarity:.2f}")
col4.metric("Urgent Issues", len(filtered_df[filtered_df['polarity'] < -0.5]), help="Tickets with high negative intensity")

st.divider()

# 5. Visualizations
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Sentiment Distribution")
    color_map = {'Positive': '#28a745', 'Negative': '#ff4b4b', 'Neutral': '#b2b2b2'}
    counts = filtered_df['sentiment'].value_counts()
    colors = [color_map.get(label, '#b2b2b2') for label in counts.index]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor': 'white'})
    ax1.axis('equal') 
    st.pyplot(fig1)

with row1_col2:
    st.subheader("Sentiment by Category")
    # Pivot for stacked bar chart
    cat_sentiment = filtered_df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
    fig2, ax2 = plt.subplots()
    cat_sentiment.plot(kind='bar', stacked=True, ax=ax2, color=['#ff4b4b', '#b2b2b2', '#28a745'])
    plt.xticks(rotation=45)
    st.pyplot(fig2)

st.subheader("Sentiment Trend (Oct - Dec 2025)")
# Resample to daily counts
trend_df = filtered_df.set_index('timestamp').resample('D')['sentiment'].value_counts().unstack(fill_value=0)
st.line_chart(trend_df, color=["#ff4b4b", "#b2b2b2", "#28a745"])

# 6. Raw Data & Insights
with st.expander("View Full Complaint Records"):
    st.dataframe(filtered_df[['timestamp', 'category', 'ticket_text', 'sentiment', 'polarity']], use_container_width=True)

# Footer
st.markdown("---")
st.caption("Developed by Ifedayo Ayomide Daniel | Matric No: 24280599009 | LASU Data Science")