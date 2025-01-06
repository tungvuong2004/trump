import tweepy
import pandas as pd
import time

# Step 1: Set up your Twitter API credentials
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIHFxwEAAAAAe7dEh8CExcSC6qd3Gdgw8FqBRRE%3DaLGYNTcOHKwqmoXCvCK1LfU8B9oPIyUQqq7vVq97dkFkPKYFrz"  # Replace this with your actual bearer token

if not BEARER_TOKEN:
    raise ValueError("Bearer token is missing. Please provide a valid bearer token.")

# Step 2: Authenticate with the Twitter API v2
try:
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    print("Successfully authenticated with Twitter API.")
except Exception as e:
    raise RuntimeError(f"Authentication failed: {e}")

def fetch_tweets_v2(query, max_results=100, total_tweets=100):
    """
    Fetch recent tweets given a query, with optimized rate-limit handling.
    """
    tweets = []
    next_token = None
    retries = 0

    while len(tweets) < total_tweets:
        try:
            # Fetch tweets
            response = client.search_recent_tweets(
                query=query,
                tweet_fields=["created_at", "text"],
                max_results=max_results,
                next_token=next_token
            )

            # Check if there are tweets in the response
            if response.data:
                for tweet in response.data:
                    tweets.append({"Tweet": tweet.text, "Created_at": tweet.created_at})
                    if len(tweets) >= total_tweets:
                        break

                next_token = response.meta.get("next_token")
                if not next_token:
                    print("No more tweets available.")
                    break

            print(f"Fetched {len(tweets)} tweets so far...")

        except tweepy.errors.TooManyRequests as e:
            # Handle rate limit dynamically
            reset_time = e.response.headers.get("x-rate-limit-reset")
            if reset_time:
                wait_time = max(0, int(reset_time) - int(time.time()))
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                backoff = min(15 * (2 ** retries), 15 * 60)  # Cap at 15 minutes
                print(f"Rate limit hit. Backoff for {backoff} seconds.")
                time.sleep(backoff)
                retries += 1

        except tweepy.errors.Unauthorized as e:
            print("Error: Unauthorized. Please check your BEARER_TOKEN.")
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return tweets

if __name__ == "__main__":
    print("Fetching tweets for Donald Trump...")
    trump_query = "Donald Trump -is:retweet"
    trump_tweets = fetch_tweets_v2(query=trump_query, max_results=100, total_tweets=100)

    if trump_tweets:
        pd.DataFrame(trump_tweets).to_csv("new_trump_tweets.csv", index=False)
        print("Saved tweets about Donald Trump to 'new_trump_tweets.csv'")

    print("Tweets collection completed!")

