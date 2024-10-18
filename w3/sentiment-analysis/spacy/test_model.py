import spacy
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import re
import kagglehub
from collections import Counter
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the trained sentiment model
nlp = spacy.load("sentiment_model")

# Load the dataset
path = kagglehub.dataset_download(
    "niraliivaghani/flipkart-product-customer-reviews-dataset")
print("Path to dataset files:", path)
data = pd.read_csv(path + '/Dataset-SA.csv', header=None, names=[
                   "product_name", "product_price", "Rate", "Review", "Summary", "Sentiment"], nrows=10)
data = data.dropna(subset=["Summary", "Sentiment"], axis=0)


def preprocess_tweet(tweet):
    # print(tweet)
    tweet = tweet.lower()
    # Remove links and special characters
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove all non-letter characters
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    # print("ALTERED: " + tweet.strip())
    return tweet.strip()


# Apply preprocessing
data['Summary'] = data['Summary'].apply(preprocess_tweet)

# Prepare true labels
true_labels = data['Sentiment'].apply(lambda x: "positive" if x == "positive" else (
    "negative" if x == "negative" else "neutral")).tolist()

# Predict sentiments
predicted_labels = []
for text in data['Summary']:
    doc = nlp(text)
    # Get the label with the highest score
    predicted_sentiment = max(doc.cats, key=doc.cats.get)
    predicted_labels.append(predicted_sentiment)


# Calculate accuracy and other metrics
print("True label distribution:", Counter(true_labels))
print("Predicted label distribution:", Counter(predicted_labels))
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print(classification_report(true_labels, predicted_labels,
      target_names=["negative", "neutral", "positive"]))
