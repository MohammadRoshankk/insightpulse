
import pandas as pd
from transformer_sentiment import transformer_label
from score_sentiment_rating import score_sentiment_rating

df = pd.read_csv("data/sample_reviews.csv")

df = df.dropna(subset=["text"])

def rating_to_label(rating):
    if rating >= 4:
        return "positive"
    elif rating <=2:
        return "negative"
    else:
        return "neutral"


correct = 0
total = 0

for index,row in df.iterrows():
    true_label = rating_to_label(row["rating"])

    if true_label =="neutral":
        continue

    predicted_label = transformer_label(row["text"])


    if predicted_label == true_label:
        correct +=1
    total +=1


accuracy = correct/total
print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")