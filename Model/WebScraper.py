import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import pymongo
from openai import OpenAI
from StopWordsRemoval import remove_stop_words
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# MongoDB setup
client = pymongo.MongoClient("mongodb+srv://julionmalek01:Darkx246!*@cluster0.xzvg5pm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["echoescape"]
collection = db["articles"]

# OpenAI API setup
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# List of sources to scrape
sources = [
    {"name": "Fox News", "base_url": "https://www.foxnews.com", "pattern": r'https://www\.foxnews\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'},
    {"name": "Al Jazeera", "base_url": "https://www.aljazeera.com", "pattern": r'https://www\.aljazeera\.com/news/[a-zA-Z0-9-]+'},
    {"name": "CNN", "base_url": "https://www.cnn.com", "pattern": r'https://www\.cnn\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'},
    {"name": "Daily Express", "base_url": "https://www.express.co.uk", "pattern": r'https://www\.express\.co\.uk/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'},
    {"name": "msnbc", "base_url": "https://www.msnbc.com", "pattern": r'https://www\.msnbc\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'},
    {"name": "CBC", "base_url": "https://www.cbc.ca", "pattern": r'https://www\.cbc\.ca/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'},


    # Add more sources here
]

def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

def extract_article_links(homepage_html, base_url, pattern):
    soup = BeautifulSoup(homepage_html, 'html.parser')
    links = soup.find_all('a', href=True)
    article_links = []
    for link in links:
        href = link['href']
        # Ensure the URL is absolute
        if href.startswith('/'):
            href = f"{base_url.rstrip('/')}{href}"
        elif not href.startswith('http'):
            href = f"{base_url.rstrip('/')}/{href}"
        # Filter to only include article links based on URL pattern
        if re.match(pattern, href):
            if href not in article_links:
                article_links.append(href)
    return article_links

def fetch_article_data(url):
    html = fetch_page(url)
    if not html:
        return None
    title = extract_title(html)
    date = datetime.now().strftime('%Y-%m-%d')  # Use current date for now
    raw_text = clean_html(html)
    cleaned_text = clean_text(raw_text)
    text_with_stop_words = cleaned_text  # Keep the text with stop words for bias detection
    text_without_stop_words = remove_stop_words(cleaned_text)  # Remove stop words for storing in the database
    return {
        "title": title,
        "date": date,
        "text_with_stop_words": text_with_stop_words,
        "text_without_stop_words": text_without_stop_words,
        "url": url
    }

def extract_title(html):
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.get_text(strip=True) if title_tag else 'No Title Found'

def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
        element.decompose()
    return soup.get_text(separator=' ')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()

def detect_bias(text):
    allowed_biases = {"far-right", "right", "right-leaning", "center", "left-leaning", "left", "far-left"}
    try:
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze the following text and determine its political bias (e.g., left, right, center) but write only one word and the words are limited to far-right, right, right-leaning, center, left-leaning, left, far-left. DO NOT WRITE ANYTHING ELSE, KEEP IT ALL LOWERCASE, DO NOT WRITE 'RIGHT-WING' or 'right-wing' just the 7 options I gave you! I need this to be able to train the model, also look at the source it can tell you a lot about the lean use that to help you a bit, I need even numbers of all leans to train my model:\n\n{text}"
                }
            ],
            model="gpt-3.5-turbo",
        )
        bias = response.choices[0].message.content.strip()
        if bias in allowed_biases:
            return bias
        else:
            logging.warning(f"Invalid bias detected: {bias}")
            return None
    except Exception as e:
        logging.error(f"Error detecting bias: {e}")
        return None

def save_article_to_mongo(article_data):
    if collection.find_one({"url": article_data["url"]}):
        logging.info(f"Article with URL {article_data['url']} already exists in the database.")
    else:
        collection.insert_one({
            "title": article_data["title"],
            "date": article_data["date"],
            "text": article_data["text_without_stop_words"],
            "url": article_data["url"],
            "bias": article_data["bias"],
            "source": article_data["source"]
        })
        logging.info(f"Article with URL {article_data['url']} added to the database.")

def clear_collection():
    result = collection.delete_many({})
    logging.info(f"Deleted {result.deleted_count} documents from the collection.")

def main():
    #clear_collection()
    
    for source in sources:
        base_url = source["base_url"]
        pattern = source["pattern"]
        homepage_url = base_url
        
        homepage_html = fetch_page(homepage_url)
        if not homepage_html:
            logging.error(f"Failed to fetch homepage HTML for {homepage_url}")
            continue

        article_links = extract_article_links(homepage_html, base_url, pattern)
        logging.info(f"Found {len(article_links)} article links on {source['name']}")

        for link in article_links:
            try:
                article_data = fetch_article_data(link)
                if article_data:
                    article_data['bias'] = detect_bias(article_data['text_with_stop_words'])
                    if article_data['bias']:  # Only save if bias is valid
                        article_data['source'] = source["name"]
                        save_article_to_mongo(article_data)
                    else:
                        logging.warning(f"Discarding article with invalid bias: {link}")
                else:
                    logging.warning(f"No article data extracted for {link}")
            except Exception as e:
                logging.error(f"Failed to process {link}: {e}")

if __name__ == '__main__':
    main()
