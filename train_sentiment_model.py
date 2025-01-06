import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
import joblib

# Step 1: Load the dataset
data = pd.read_csv('sentiment_trump_tweets.csv')  # Replace with your training dataset
print("Training Dataset Preview:")
print(data.head())

# Ensure dataset contains 'Tweet' and 'Sentiment' columns
if 'Tweet' not in data.columns or 'Sentiment' not in data.columns:
    raise ValueError("Dataset must contain 'Tweet' and 'Sentiment' columns.")

# Step 2: Preprocess the data
X = data['Tweet']       # Features: Tweets
y = data['Sentiment']   # Target: Sentiments (Positive, Neutral, Negative)

# Display class distribution
print("\nClass distribution in training dataset:")
print(data['Sentiment'].value_counts())

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=5000, 
    ngram_range=(1, 2)  # Include bi-grams for richer features
)
X_vec = vectorizer.fit_transform(X)

# Handle class imbalance with oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_vec, y)

print("\nClass distribution after oversampling:")
print(pd.Series(y_resampled).value_counts())

# Step 3: Hyperparameter Tuning with GridSearchCV
params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}  # Regularization parameter tuning
clf = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42), 
    param_grid=params, 
    cv=5, 
    scoring='accuracy'
)
clf.fit(X_resampled, y_resampled)

print("\nBest Parameters from GridSearchCV:", clf.best_params_)

# Step 4: Evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
y_pred = clf.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Debugging: Check predictions on test set
print("\nDebugging Predictions on Test Set:")
for i in range(5):  # Show 5 examples from the test set
    print(f"Tweet: {X.iloc[i]}")
    print(f"Actual Sentiment: {y.iloc[i]}, Predicted Sentiment: {y_pred[i]}")

# Step 5: Save the model and vectorizer
joblib.dump(clf, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nModel and vectorizer saved to 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl'")
