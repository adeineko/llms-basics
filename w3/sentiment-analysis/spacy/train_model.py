import spacy
from spacy.training import Example
import random
import warnings
import pandas as pd
import kagglehub
import re

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load a blank English model
nlp = spacy.blank("en")
# Create a text classifier
text_classifier = nlp.add_pipe("textcat", last=True)
text_classifier.add_label("positive")
text_classifier.add_label("negative")
text_classifier.add_label("neutral")

# Load the dataset
path = kagglehub.dataset_download(
    "niraliivaghani/flipkart-product-customer-reviews-dataset")
print("Path to dataset files:", path)
data = pd.read_csv(path + '/Dataset-SA.csv', header=None, names=[
                   "product_name", "product_price", "Rate", "Review", "Summary", "Sentiment"], nrows=2000)
data = data.dropna(subset=["Summary", "Sentiment"], axis=0)


def preprocess_tweet(tweet):
    # print(tweet)
    tweet = tweet.lower()
    # Remove links and special characters
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)  # Remove mentions and hashtags
    # Remove all non-letter characters
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    # print("ALTERED: " + tweet.strip())
    return tweet.strip()


# Apply preprocessing
data['Summary'] = data['Summary'].apply(preprocess_tweet)


def map_sentiment(target):
    # print(target)
    if target == "negative":
        # print("- negative")
        return {"cats": {"negative": 1, "positive": 0, "neutral": 0}}
    elif target == "neutral":
        # print("- neutral")
        return {"cats": {"negative": 0, "positive": 0, "neutral": 1}}
    elif target == "positive":
        # print("- positive")
        return {"cats": {"negative": 0, "positive": 1, "neutral": 0}}
    else:
        return None


# Prepare the training data
training_data = []
for index, row in data.iterrows():
    sentiment = map_sentiment(row['Sentiment'])
    if sentiment is not None:  # Ensure we only include valid sentiments
        training_data.append((row['Summary'], sentiment))

random.shuffle(training_data)
nlp.initialize()

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    random.shuffle(training_data)
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5)

# Save the model
nlp.to_disk("sentiment_model")
