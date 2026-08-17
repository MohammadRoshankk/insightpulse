def score_sentiment(text):
    poscount=0
    negcount=0
    positive_words = ["great", "good", "excellent", "fast", "amazing"]
    negative_words = ["bad", "poor", "terrible", "slow", "disappointed"]
    for word in positive_words:
        if word in text:
            poscount+=1
    #return poscount
    for word in negative_words:
        if word in text:
            negcount+=1
    #return negcount
    
    if poscount>negcount:
        return "positive"
    elif negcount>poscount:
        return "negative"
    else:
        return "neutral"    
    
#print(score_sentiment("the service was great and fast"))
#print(score_sentiment("terrible experience, broke immediately"))
#print(score_sentiment("it was okay I guess"))
#print(score_sentiment("the delivery was not bad at all"))
#print(score_sentiment("this is a fantastic product for my classroom"))

