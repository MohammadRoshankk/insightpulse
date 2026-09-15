from executive_summary import batch_summarize
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
from score_sentiment import score_sentiment
from score_sentiment_rating import score_sentiment_rating
from vader_sentiment import vader_label,vader_label_custom
from clean_text import clean_text
from collections import Counter as C
from nltk.corpus import stopwords
from keyword_extraction import get_bigrams
stop_words = set(stopwords.words("english"))
from tfidf_keywords import get_top_tfidf_words
from detect_category import detect_category
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from executive_summary import generate_summary,final_summary












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
df["clean_text"] = df["text"].apply(clean_text)




df["category"] = df["clean_text"].apply(detect_category)
print(df["category"].value_counts())
print(df[df["category"]=="complaint"]["text"].head(10))






df["date"] = pd.to_datetime(df["date"])
# print(df["date"].dtype)
# print(df["date"].max())
# print(df["date"].min())


df["month"] = df["date"].dt.to_period("M")

monthly_avg_rating = df.groupby("month")["rating"].mean()
print(monthly_avg_rating)


monthly_avg_rating.plot(kind="line", marker= "o")
plt.title("Average monthly rating")
plt.xlabel("Month")
plt.ylabel("Average Rating")
plt.show()




monthly_counts = df.groupby("month")["review_id"].count()
print(monthly_counts)








all_words=[]
for text in df["clean_text"]:
    words = text.split()
    for word in words:
        if word not in stop_words:
            all_words.append(word)
word_count=C(all_words)
print(word_count.most_common(15))



all_bigrams=[]
for text in df["clean_text"]:
    words =[]
    for w in text.split():
        if w not in stop_words:
            words.append(w)
    clean_sentance = " ".join(words)
    bigrams = get_bigrams(clean_sentance)

    for i in bigrams:
        all_bigrams.append(i)

bigram_count = C(all_bigrams)
print(bigram_count.most_common(15))


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


top_words = get_top_tfidf_words(df["clean_text"].tolist())
print(top_words)





# GENERATE EXECTUIVE SUMMARY

sample_reviews = "\n".join(df["text"].head(250).tolist())
# summary = generate_summary(sample_reviews)
# print ("Summary of the reviews are : " +"\n" ,summary)
all_reviews = df["clean_text"].tolist()
batch_summary = batch_summarize(all_reviews,batch_size=50)
print("--- BATCH SUMMARIES ---")

for i ,s in enumerate(batch_summary):
    print(f"Batch {i+1}: {s}\n")

print("--- FINAL SUMMARY ---")
print(final_summary(batch_summary))
