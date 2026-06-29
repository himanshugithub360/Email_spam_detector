import pickle

tfidf = pickle.load(open("vectorizer.pkl", "rb"))

print(type(tfidf))
print(hasattr(tfidf, "vocabulary_"))