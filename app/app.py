import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

st.set_page_config(page_title=Phlex Sentiment Dashboard, layout=wide)

st.title(📊 Phlex Payment Customer Sentiment Analysis)
st.markdown(Analysis of transaction complaints from Oct - Dec 2025.)

# Load Data
df = pd.read_csv('..datacomplaints.csv')

# Sentiment Logic
def get_sentiment(text)
    score = TextBlob(text).sentiment.polarity
    return 'Positive' if score  0 else ('Negative' if score  0 else 'Neutral')

df['sentiment'] = df['ticket_text'].apply(get_sentiment)

# Sidebar Filters
category = st.sidebar.multiselect(Filter by Category, options=df['category'].unique(), default=df['category'].unique())
filtered_df = df[df['category'].isin(category)]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric(Total Tickets, len(filtered_df))
col2.metric(Negative Sentiment, f{len(filtered_df[filtered_df['sentiment']=='Negative'])})
col3.metric(Positive Sentiment, f{len(filtered_df[filtered_df['sentiment']=='Positive'])})

# Visuals
st.subheader(Sentiment Distribution)
fig, ax = plt.subplots()
filtered_df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff','#99ff99'])
st.pyplot(fig)

st.subheader(Raw Ticket Data)
st.dataframe(filtered_df)