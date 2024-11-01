import pandas as pd
import spacy
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nlp = spacy.load("sentiment_model")

# Load the data
data = pd.read_csv('../SentimentAssignmentReviewCorpus.csv')
data = data.dropna(subset=["reviewBody", "reviewTitle"],
                   axis=0).astype(str)


def extract_aspects(text):
    """Extracts aspects from the text using SpaCy."""
    aspects = set()
    doc = nlp(text)
    # for token in doc:
    #     # Identify nouns and proper nouns as potential aspects
    #     if token.pos_ in ["NOUN", "PROPN"]:
    #         aspects.add(token.text)
    # Identify nouns and adjectives related to nouns (potential aspects)
    for token in doc:
        # Subject, Direct Object, Attribute, or Subject of Passive
        if token.dep_ in ["nsubj", "dobj", "attr", "nsubjpass"]:
            aspects.add(token.text)
        elif token.dep_ == "amod":  # Adjective modifier
            aspects.add(token.head.text)  # Add the head noun of the adjective

    return list(aspects)


def get_aspect_sentiment(text, aspect):
    if isinstance(text, str) and aspect:
        # Formatting for aspect analysis
        aspect_sentence = f"{text} [SEP] {aspect}"
        doc = nlp(aspect_sentence)
        predicted_sentiment = max(doc.cats, key=doc.cats.get)
        return predicted_sentiment
    return 'neutral'


def calculate_overall_sentiment(sentiments):
    if not sentiments:
        return 'neutral'

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    # Count occurrences of each sentiment
    for sentiment in sentiments:
        sentiment_counts[sentiment] += 1

    # Determine overall sentiment based on counts
    if sentiment_counts['positive'] > sentiment_counts['negative'] and sentiment_counts['positive'] > sentiment_counts['neutral']:
        return 'positive'
    elif sentiment_counts['negative'] > sentiment_counts['positive'] and sentiment_counts['negative'] > sentiment_counts['neutral']:
        return 'negative'
    else:
        return 'neutral'


# Process reviews to extract aspects and their sentiments
aspect_sentiments = []

for _, row in data.iterrows():
    title = row['reviewTitle']
    body = row['reviewBody']

    title_aspects = extract_aspects(title)
    body_aspects = extract_aspects(body)

    title_sentiments = {aspect: get_aspect_sentiment(
        title, aspect) for aspect in title_aspects}
    body_sentiments = {aspect: get_aspect_sentiment(
        body, aspect) for aspect in body_aspects}

    # Calculate overall sentiment for title and body
    overall_sentiment_title = calculate_overall_sentiment(
        list(title_sentiments.values()))
    overall_sentiment_body = calculate_overall_sentiment(
        list(body_sentiments.values()))

    aspect_sentiments.append({
        'reviewTitle': title,
        'reviewBody': body,
        'title_sentiment': overall_sentiment_title,
        'body_sentiment': overall_sentiment_body,
        'title_aspects': title_aspects,
        'title_sentiments': title_sentiments,
        'body_aspects': body_aspects,
        'body_sentiments': body_sentiments,
    })

# Create a DataFrame from the results
aspect_sentiments_df = pd.DataFrame(aspect_sentiments)

# Save the results
aspect_sentiments_df.to_csv('predicted_aspect_sentiments.csv', index=False)

# Display the results
print(aspect_sentiments_df[['reviewTitle', 'reviewBody',
      'title_sentiment', 'body_sentiment', 'title_aspects', 'title_sentiments', 'body_aspects', 'body_sentiments']])
