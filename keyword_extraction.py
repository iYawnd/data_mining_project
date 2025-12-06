from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from helpers import is_valid_keyword

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# Input: preprocessed text
# Output: list of keywords

# KeyBERT
# deep learning model
# https://maartengr.github.io/KeyBERT/index.html
def keybert_extract_keywords(text, top_n=5, diversity=0.7):
    if not text: return []

    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1,1),   # only single word keywords
        stop_words='english',
        use_mmr=True,   # maximal marginal relevance
        diversity=diversity,
        top_n=top_n * 2   # number of keywords to return, doubled because filter might delete some
    )

    filtered_keywords = []
    # filter
    for kw in keywords:
        # get word
        word = kw[0]

        # keyword must be noun, proper noun, or number
        if is_valid_keyword(word):
            filtered_keywords.append(word)
        # stop when enough keywords are found
        if len(filtered_keywords) >= top_n:
            break

    # return list of words (don't care about exact relevance scores)
    return filtered_keywords


# TF-IDF
# statistical model
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
def tfidf_extract_keywords(text, top_n=5):
    if not text: return []

    # convert text to matrix of numbers
    vectorizer=TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,1)
    )

    try:
        # create frequency matrix
        tfidf_matrix = vectorizer.fit_transform([text])
    except ValueError:
        # fails if text is empty or contains only stopwords
        return []

    # get list of words
    feature_names = vectorizer.get_feature_names_out()

    # format scores to list
    scores = tfidf_matrix.toarray()[0]

    # list of indices from best to worst words
    sorted_idx = np.argsort(scores)[::-1]
    
    # get list of top N words with filter
    top_keywords = []
    for i in sorted_idx:
        word = feature_names[i]

        if is_valid_keyword(word):
            top_keywords.append(word) 

        if len(top_keywords) >= top_n:
            break

    return top_keywords