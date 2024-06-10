import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import joblib
from lime.lime_text import LimeTextExplainer
import numpy as np

# Read the CSV file
df = pd.read_csv('labeled_articles.csv')

print(df['label'].value_counts())


# Split the data into training and validation sets with stratified split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Create a pipeline with TF-IDF vectorization and Logistic Regression
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Define hyperparameters for GridSearch
param_grid = {
    'tfidfvectorizer__max_features': [5000, 10000],
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    'logisticregression__C': [0.1, 1, 10]
}

# Perform GridSearch to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(train_texts, train_labels)

# Print the best parameters found by GridSearch
print(f"Best parameters: {grid_search.best_params_}")

# Validate the best model
best_model = grid_search.best_estimator_
predicted_labels = best_model.predict(val_texts)

# Evaluate the model with zero_division set to 1
print(metrics.classification_report(val_labels, predicted_labels, zero_division=1))

# Save the best model
joblib.dump(best_model, 'best_political_bias_model.pkl')

# Optional: Load and use the model
model = joblib.load('best_political_bias_model.pkl')
texts = ["On the morning of October 7, after waking to sirens signaling rocket attacks from Gaza, photojournalist Ziv Koren drove south on his motorcycle to capture the unfolding horrors of the Hamas massacre."]
predicted_labels = model.predict(texts)
print(predicted_labels)

# Use LIME to explain the prediction
explainer = LimeTextExplainer(class_names=['Pro-Israel', 'Pro-Palestinian', 'Center'])
idx = 0  # index of the text you want to explain
exp = explainer.explain_instance(texts[idx], model.predict_proba, num_features=10)

# Print the explanation
print(exp.as_list())

# Visualize the explanation
exp.show_in_notebook(text=True)
