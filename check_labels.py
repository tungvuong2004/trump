import pandas as pd

# Load the dataset
data = pd.read_csv('sentiment_trump_tweets.csv')  # Replace with your dataset's actual path

# Check the distribution of sentiment labels
print(data['Sentiment'].value_counts())

