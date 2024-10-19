import pandas as pd
import spacy
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("sentiment_model")


def get_sentiment(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Predict sentiments
        doc = nlp(text)
        # Get the label with the highest score
        predicted_sentiment = max(doc.cats, key=doc.cats.get)
        return predicted_sentiment
    return 'neutral'  # Return neutral if the input is not a string or is NaN


# Apply sentiment analysis to reviewTitle and reviewBody
data = pd.read_csv('../SentimentAssignmentReviewCorpus.csv')
data['title_sentiment'] = data['reviewTitle'].apply(get_sentiment)
data['body_sentiment'] = data['reviewBody'].apply(get_sentiment)


# Display and save the results
data.to_csv('predicted_sentiments.csv', index=False)
print(data[['reviewTitle', 'title_sentiment', 'reviewBody', 'body_sentiment']])
