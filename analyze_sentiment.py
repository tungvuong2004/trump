from textblob import TextBlob
import pandas as pd

# File paths
file_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\cleaned_trump_tweets.csv"
output_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\sentiment_trump_tweets.csv"

# Load the cleaned tweets
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Ensure the file exists.")
    exit()

# Check if the necessary column exists
if "Cleaned_Tweet" not in df.columns:
    print("Error: 'Cleaned_Tweet' column not found in the dataset.")
    exit()

# Drop rows where "Cleaned_Tweet" is NaN
df = df.dropna(subset=["Cleaned_Tweet"])

# Ensure all values in "Cleaned_Tweet" are strings
df["Cleaned_Tweet"] = df["Cleaned_Tweet"].astype(str)

# Function to analyze sentiment
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
print("Analyzing sentiments...")
df["Sentiment"] = df["Cleaned_Tweet"].apply(analyze_sentiment)

# Save the results to a new CSV file
df.to_csv(output_path, index=False)
print(f"Sentiment analysis completed. Results saved to {output_path}")
