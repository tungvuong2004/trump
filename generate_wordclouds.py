import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the sentiment data
file_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\sentiment_trump_tweets.csv"

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Ensure the file exists.")
    exit()

# Check if the necessary columns exist
if "Sentiment" not in df.columns or "Cleaned_Tweet" not in df.columns:
    print("Error: Required columns ('Sentiment' or 'Cleaned_Tweet') are missing in the dataset.")
    exit()

# Generate word clouds for each sentiment
for sentiment in ["Positive", "Negative", "Neutral"]:
    print(f"Generating word cloud for {sentiment} tweets...")
    
    # Combine all tweets of the current sentiment
    tweets = " ".join(df[df["Sentiment"] == sentiment]["Cleaned_Tweet"])
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(tweets)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment} Tweets Word Cloud")
    plt.show()
