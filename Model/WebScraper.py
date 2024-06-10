from flask import Flask, render_template, request, redirect, url_for
import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import re
from StopWordsRemoval import remove_stop_words
import csv

# Download the punkt package for tokenization
nltk.download('punkt')

app = Flask(__name__)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

def fetch_article(url):
    # Fetch the article
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.text

def clean_html(html):
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Remove ads, photos, and other unwanted elements
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
        element.decompose()

    # Extract text content
    text = soup.get_text(separator=' ')
    return text

def clean_text(text):
    # Remove unwanted characters and multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)  # Remove content within brackets
    text = text.strip()
    return text

def label_and_save(text, label, output_file):
    # Create a DataFrame and append to the CSV file
    df = pd.DataFrame([[text, label]], columns=['text', 'label'])
    df.to_csv(output_file, mode='a', index=False, header=False, quoting=csv.QUOTE_ALL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_article', methods=['POST'])
def add_article():
    url = request.form['url']
    label = request.form['label']
    output_file = 'labeled_articles.csv'

    # Fetch and clean the article
    html = fetch_article(url)
    raw_text = clean_html(html)
    cleaned_text = clean_text(raw_text)
    stop_words_removed = remove_stop_words(cleaned_text)

    # Label and save the data
    label_and_save(stop_words_removed, label, output_file)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
