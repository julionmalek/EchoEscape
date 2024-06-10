# StopWordsRemoval.py

import nltk
from nltk.corpus import stopwords
import string

# Ensure you have downloaded the stopwords
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    # Remove punctuation and convert to lower case
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Split text into words and remove stop words
    cleaned_text = ' '.join([word for word in text.split() if word not in stop_words])
    return cleaned_text
