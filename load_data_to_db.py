import pandas as pd
from database import SessionLocal , Review


df = pd.read_csv("data/sample_reviews.csv")
#df["text"] = df["text"].fillna("No Review")
df = df.dropna(subset=["text"])
df = df.dropna(subset=["rating"])
df["date"] = pd.to_datetime(df["date"])

session = SessionLocal()

for _, row in df.iterrows():
    review = Review(
        text=row["text"],
        rating=row["rating"],
        date=row["date"]
    )
    session.add(review)


session.commit()
session.close()

print(f"Loaded {len(df)} reviews into the database.")
