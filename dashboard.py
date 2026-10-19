import pandas as pd
import streamlit as st
from database import SessionLocal, Review
st.title("InsightPulse — Customer Review Intelligence")

session = SessionLocal()
reviews = session.query(Review).all()
session.close()

df = pd.DataFrame([{
    "text": r.text,
    "rating": r.rating,
    "date": r.date,
    "sentiment": r.sentiment,
    "category": r.category
} for r in reviews])

df["date"] = pd.to_datetime(df["date"])

st.metric("Total Reviews", len(df))
st.metric("Average Rating", round(df["rating"].mean(), 2))

st.subheader("Sentiment Breakdown")
sentiment_counts = df["sentiment"].value_counts()
st.bar_chart(sentiment_counts)

df["month"] = df["date"].dt.to_period("M").astype(str)
monthly_rating = df.groupby("month")["rating"].mean()

st.subheader("Average Rating Over Time")
st.line_chart(monthly_rating)

st.subheader("Review Categories")
category_counts = df["category"].value_counts()
st.bar_chart(category_counts)

st.subheader("Sample Complaints")
complaints = df[df["category"] == "complaint"]["text"].head(10)
st.write(complaints)