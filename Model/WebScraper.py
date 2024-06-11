from datetime import datetime
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

def label_and_save_to_mongo(title, text, labels, funding, url, date):

    
    # Create a document and insert into MongoDB
    document = {
        "title": title,
        "text": text,
        "labels": labels,
        "funding": funding,
        "link": url,
        "date": date

    }
    
    # Check for duplicates before inserting
    if collection.find_one({"link": url}):
        print(f"Article with URL {url} already exists in the database.")
    else:
        collection.insert_one(document)
        print(f"Article with URL {url} added to the database.")

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

    #add date of today
    date = datetime.now().strftime('%Y-%m-%d')

    raw_text = clean_html(html)
    cleaned_text = clean_text(raw_text)
    stop_words_removed = remove_stop_words(cleaned_text)

    # Label and save the data
    label_and_save_to_mongo(title, stop_words_removed, labels, funding, url, date)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
