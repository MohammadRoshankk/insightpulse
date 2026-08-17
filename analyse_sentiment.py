import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
from score_sentiment import score_sentiment
from score_sentiment_rating import score_sentiment_rating
from vader_sentiment import vader_label,vader_label_custom


df = pd.read_csv("data/sample_reviews.csv")
df["text"]=df["text"].fillna("No review")
df=df.dropna(subset=["rating"])
df["rating"]=df["rating"].astype(int)
df["text"]=df["text"].str.strip().str.lower()
df["predicted_sentiment_text"]=df["text"].apply(score_sentiment)
df["predicted_sentiment_rating"]=df["rating"].apply(score_sentiment_rating)
print(df[["text","rating","predicted_sentiment_text","predicted_sentiment_rating"]].head(10))     
df["match"]=df["predicted_sentiment_text"]==df["predicted_sentiment_rating"]
df["vader_label"] = df["text"].apply(vader_label)
df["match_laber"] = df["vader_label"] == df["predicted_sentiment_rating"]



print("Rule-based-accuracy: ",df["match"].mean())
print("vader Accuracy: ", df["match_laber"].mean())



print(df["match_laber"].mean())
#print(df["match"].value_counts())
#print(df["match"].mean())
#print(df.groupby("predicted_sentiment_rating")["match"].mean())
print(df["vader_label"].head(10))
print(df.groupby("predicted_sentiment_rating")["match_laber"].mean())

for t in [0.05, 0.1, 0.2, 0.3]:
    df["temp_label"]=df["text"].apply(lambda x: vader_label_custom(x,threshold=t))
    accuracy=(df["temp_label"] == df["predicted_sentiment_rating"]).mean()
    print(f"Threshold {t} : accuracy = {accuracy}")