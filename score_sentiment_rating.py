def score_sentiment_rating(rating):
    if rating == 4 or rating == 5:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"
