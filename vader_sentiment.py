from nltk.sentiment import SentimentIntensityAnalyzer

sia =  SentimentIntensityAnalyzer()

def vader_label(text):
    score=sia.polarity_scores(text)["compound"]
    if score>=0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def vader_label_custom(text,threshold=0.05):
    score=sia.polarity_scores(text)["compound"]
    if score>=threshold:
        return "positive"
    elif score<=-threshold:
        return "negative"
    else:
        return "neutral"
        