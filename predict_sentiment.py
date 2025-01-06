import joblib

# Step 1: Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Step 2: Input new tweets for sentiment prediction
new_tweets = [
    "The policies are amazing and will lead to great success.",
    "I hate the current state of affairs in the country.",
    "This is a neutral tweet without much emotion."
]

# Step 3: Preprocess and transform the new tweets
new_tweets_vec = vectorizer.transform(new_tweets)

# Step 4: Predict sentiments
predictions = model.predict(new_tweets_vec)

# Step 5: Display the results
print("\nPredicted Sentiments:")
for tweet, sentiment in zip(new_tweets, predictions):
    print(f"Tweet: \"{tweet}\" -> Sentiment: {sentiment}")
