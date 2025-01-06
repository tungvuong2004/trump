import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the dataset
file_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\sentiment_trump_tweets_vader.csv"
df = pd.read_csv(file_path)

# Check if necessary columns exist
if "Sentiment" not in df.columns or "VADER_Sentiment" not in df.columns:
    print("Error: One or both sentiment columns are missing.")
    exit()

# Check available columns and define the correct text column
print("Available columns:", df.columns)
text_column = "Cleaned_Tweet"  # Use the cleaned text column for analysis

if text_column not in df.columns:
    raise KeyError(f"Column '{text_column}' not found in the DataFrame.")

# Replace non-string values with an empty string
df[text_column] = df[text_column].fillna("").astype(str)

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Apply VADER to calculate sentiment scores
def analyze_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return pd.Series({
        "Positive": scores["pos"],
        "Neutral": scores["neu"],
        "Negative": scores["neg"],
        "Compound": scores["compound"]
    })

# Add VADER sentiment scores to the DataFrame
df_vader_scores = df[text_column].apply(analyze_vader_sentiment)
df = pd.concat([df, df_vader_scores], axis=1)

# Compare TextBlob and VADER sentiments
comparison = df[["Sentiment", "VADER_Sentiment"]].value_counts()
print("Comparison between TextBlob and VADER Sentiments:")
print(comparison)

# Calculate agreement rate
df["Agreement"] = df["Sentiment"] == df["VADER_Sentiment"]
agreement_rate = df["Agreement"].mean() * 100
print(f"Agreement Rate: {agreement_rate:.2f}%")

# Generate a confusion matrix for visualization
confusion_matrix = df.groupby(["Sentiment", "VADER_Sentiment"]).size().unstack(fill_value=0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Comparison of TextBlob and VADER Sentiments")
plt.xlabel("VADER Sentiment")
plt.ylabel("TextBlob Sentiment")
plt.show()

# Append the updated DataFrame with VADER sentiment scores to the existing CSV file
output_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\sentiment_trump_tweets_vader.csv"

try:
    # Check if the file exists
    if os.path.exists(output_path):
        # Append without writing the header
        df.to_csv(output_path, mode='a', index=False, header=False)
        print(f"Data appended successfully to '{output_path}'")
    else:
        # Write with the header if the file does not exist
        df.to_csv(output_path, index=False)
        print(f"File created and data saved to '{output_path}'")
except Exception as e:
    print(f"Error saving file: {e}")




