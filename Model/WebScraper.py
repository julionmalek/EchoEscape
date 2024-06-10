import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import re
from StopWordsRemoval import remove_stop_words
import pymongo
import csv
from flask import Flask, render_template, request, redirect, url_for

# Download the punkt package for tokenization
nltk.download('punkt')

# MongoDB setup
client = pymongo.MongoClient("mongodb+srv://julionmalek01:Darkx246!*@cluster0.xzvg5pm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["echoescape"]
collection = db["articles"]

app = Flask(__name__)

def fetch_article(url):
    # Fetch the article
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.text

def extract_title(html):
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the title tag and extract the text
    title_tag = soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else 'No Title Found'
    return title

def extract_authors(html):
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Define a function to match class names containing the word "author"
    def author_class(tag):
        if tag.has_attr('class'):
            for class_name in tag['class']:
                if 'author' in class_name.lower():
                    return True
        return False

    # Find all tags with classes containing the word "author"
    author_tags = soup.find_all(author_class)
    authors = [tag.get_text(strip=True) for tag in author_tags]
    
    # If no authors found, use a default value
    if not authors:
        print('No authors found')
        authors = ['No Authors Found']
    
    return authors

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

def label_and_save_to_mongo(title, text, labels, authors, funding, url):
    # Create a document and insert into MongoDB
    document = {
        "title": title,
        "text": text,
        "labels": labels,
        "authors": authors,
        "funding": funding,
        "link": url
    }
    collection.insert_one(document)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_article', methods=['POST'])
def add_article():
    url = request.form['url']
    labels = request.form.getlist('label')
    funding = request.form['funding']

    # Fetch and clean the article
    html = fetch_article(url)

    # Extract the title
    title = extract_title(html)

    # Extract the authors
    authors = extract_authors(html)

    raw_text = clean_html(html)
    cleaned_text = clean_text(raw_text)
    stop_words_removed = remove_stop_words(cleaned_text)

    # Label and save the data
    label_and_save_to_mongo(title, stop_words_removed, labels, authors, funding, url)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
