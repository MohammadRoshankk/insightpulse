def get_bigrams(text):
    words = text.split()
    bigram=[]
    for i in range(len(words)-1):
        pair =words[i]+" "+words[i+1]
        bigram.append(pair)
    return bigram