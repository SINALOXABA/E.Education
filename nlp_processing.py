# nlp_processing.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary resources
nltk.download('punk')
nltk.download('stopwords')


# Preprocessing essays
def preprocess_essays(essays):
    stop_words = set(stopwords.words('english'))
    preprocessed_essays = []

    for essay in essays:
        words = word_tokenize(essay.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        preprocessed_essays.append(" ".join(filtered_words))

    return preprocessed_essays


if __name__ == "__main__":
    essays = [
        "The impact of climate change on the environment is severe. We must take action.",
        "Education is the key to solving many societal problems.",
        "Technology in education helps students learn better."
    ]

    preprocessed = preprocess_essays(essays)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed)

    print("TF-IDF matrix:")
    print(tfidf_matrix.toarray())
