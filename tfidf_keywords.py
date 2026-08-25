from sklearn.feature_extraction.text import TfidfVectorizer
def get_top_tfidf_words(documents, top_n = 15):
    vectoizer = TfidfVectorizer(stop_words="english")
    matrix = vectoizer.fit_transform(documents)

    scores = matrix.mean(axis =0).tolist()[0]
    words = vectoizer.get_feature_names_out()

    word_score = list(zip(words,scores))
    word_score.sort(key=lambda x:x[1],reverse = True)



    return word_score[:top_n]