import pandas as pd
import re

# Set the correct file path
file_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\trump_tweets.csv"

# Define a function to clean tweet text
def clean_tweet_text(text):
    """
    Clean tweet text by removing URLs, mentions, hashtags, and special characters.
    """
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text) # Remove special characters
    text = re.sub(r"\s+", " ", text)    # Remove extra spaces
    return text.lower().strip()         # Convert to lowercase and strip spaces

# Try loading the CSV file
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Check if the required column exists
if "Tweet" in df.columns:
    print("Cleaning tweets...")
    df["Cleaned_Tweet"] = df["Tweet"].apply(clean_tweet_text)
else:
    print("Error: 'Tweet' column not found in the data.")
    exit(1)

# Save the cleaned data to a new CSV file
output_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\cleaned_trump_tweets.csv"
try:
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved successfully to {output_path}")
except Exception as e:
    print(f"Error saving cleaned data: {e}")
