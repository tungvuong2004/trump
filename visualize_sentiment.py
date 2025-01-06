import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sentiment data
file_path = r"C:\Users\vuong\OneDrive\Desktop\trump_sentiment_analysis\sentiment_trump_tweets.csv"

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Ensure the file exists.")
    exit()

# Check if the "Sentiment" column exists
if "Sentiment" not in df.columns:
    print("Error: 'Sentiment' column not found in the dataset.")
    exit()

# Count sentiment categories
sentiment_counts = df["Sentiment"].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()
plt.savefig("sentiment_distribution.png")
plt.figure(figsize=(8, 8))
sentiment_counts.plot(kind="pie", autopct='%1.1f%%', colors=sns.color_palette("viridis", 3))
plt.title("Sentiment Distribution")
plt.ylabel("")  # Remove y-label for a cleaner look
plt.show()
plt.savefig("sentiment_distribution_pie.png")
