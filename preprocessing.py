import re
import nltk
from nltk.corpus import stopwords

# download and load required packages
nltk.download('stopwords')
nltk.download('punkt')  # punkt: https://www.nltk.org/api/nltk.tokenize.punkt.html
nltk.download('punkt_tab')

# set stopwords
stop_words = set(stopwords.words('english'))


# Light Cleaning
# goal is to fix formatting while preserving grammar, punctuation, case, etc.
def light_clean(text):
    if not text: return ""

    # remove citations ([1], [2], etc.)
    text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)

    # fixes broken whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Deep Cleaning
# goal is to convert text to list of words
def deep_clean(text):
    if not text: return ""

    # fix formatting
    text = light_clean(text)

    # lowercase
    text = text.lower()

    # keep only words and whitespace (remove punctuation)
    text = re.sub(r'[^\w\s]', '', text)

    # remove whitespace
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]

    # return as a single string
    return " ".join(tokens)