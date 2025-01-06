import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load the dataset
file_path = 'new_trump_tweets.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Ensure the dataset contains a 'Tweet' column
if 'Tweet' not in data.columns:
    raise ValueError("The dataset must contain a 'Tweet' column.")

print("Dataset loaded. Preview:")
print(data.head())

# Step 2: Define a cleaning function
def clean_text(text):
    """
    Clean the input text by:
    - Converting to lowercase
    - Removing URLs, mentions, hashtags, and special characters
    - Removing numbers
    - Stripping extra whitespace
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", '', text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # Remove special characters and numbers
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra whitespace
    return text

# Apply the cleaning function to the 'Tweet' column
data['Cleaned_Tweet'] = data['Tweet'].apply(clean_text)

print("\nCleaned tweets preview:")
print(data[['Tweet', 'Cleaned_Tweet']].head())

# Step 3: Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')  # Replace with your trained model file
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace with your vectorizer file

# Step 4: Preprocess and predict
tweets_to_test = data['Cleaned_Tweet']
tweets_vec = vectorizer.transform(tweets_to_test)
predicted_sentiments = model.predict(tweets_vec)

# Step 5: Add predictions to the DataFrame
data['Predicted_Sentiment'] = predicted_sentiments

# Preview predictions
print("\nPredicted Sentiments:")
print(data[['Cleaned_Tweet', 'Predicted_Sentiment']].head())

# Step 6: Save the results to a new CSV file
output_file = 'tested_cleaned_tweets.csv'
data.to_csv(output_file, index=False)
print(f"\nPredictions saved to '{output_file}'")
