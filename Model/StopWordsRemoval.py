import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import csv

# Download stop words
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    # Remove punctuation and convert to lower case
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Split text into words and remove stop words
    cleaned_text = ' '.join([word for word in text.split() if word not in stop_words])
    return cleaned_text

# Read the input CSV file
input_file = 'AlJazeera.csv'
df = pd.read_csv(input_file)

# Remove stop words from the text column
df['text'] = df['text'].apply(remove_stop_words)

# Write the cleaned data to a new CSV file with proper quoting
output_file = 'data.csv'
df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

print(f"Cleaned data saved to {output_file}")
