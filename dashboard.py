from vader_sentiment import vader_label
from detect_category import detect_category as dc
import pandas as pd
import streamlit as st
st.title("InsightPulse — Customer Review Intelligence")
# st.title("This is our first dashboard")

df = pd.read_csv("data/sample_reviews.csv")
df["date"] = pd.to_datetime(df["date"])
st.metric("Total Reviews",len(df))
st.metric("Average Rating",round(df["rating"].mean(),2))

df["text"] = df["text"].fillna("No review")
df["sentiment"] = df["text"].apply(vader_label)



st.subheader("Sentiment BreakDown")
sentiment_counts = df["sentiment"].value_counts()
st.bar_chart(sentiment_counts)


df["month"] = df["date"].dt.to_period("M").astype(str)
monthly_rating =df.groupby("month")["rating"].mean()

st.subheader("Average Rating Over Time")
st.line_chart(monthly_rating)



df["category"] = df["text"].apply(dc)
st.subheader("Review Categories")
category_counts = df["category"].value_counts()
st.bar_chart(category_counts)

st.subheader("Sample Complaints")
complaints = df[df["category"]=="complaint"]["text"].head(10)
st.write(complaints)
