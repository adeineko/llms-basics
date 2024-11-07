import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, word_tokenize
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Load the data
data = pd.read_csv('../SentimentAssignmentReviewCorpus.csv')
data = data.dropna(subset=["reviewBody", "reviewTitle"], axis=0).astype(str)


def extract_aspects(text):
    """Extracts aspects from the text by identifying nouns and adjectives."""
    aspects = set()
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    for word, tag in tagged_tokens:
        # Identify nouns (NN, NNP) and adjectives (JJ) as potential aspects
        if tag in ["NN", "NNS", "NNP", "NNPS", "JJ"]:
            # Convert aspect to lowercase for consistency
            aspects.add(word.lower())
    return list(aspects)


def get_aspect_sentiment(text, aspect):
    """Uses VADER to get sentiment polarity for the given aspect within the text."""
    if isinstance(text, str) and aspect:
        # Contextualize aspect in the sentence
        aspect_text = f"{text}. Aspect: {aspect}"
        sentiment_score = vader_analyzer.polarity_scores(aspect_text)
        compound_score = sentiment_score['compound']
        # Determine sentiment label based on compound score thresholds
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    return 'neutral'


def calculate_overall_sentiment(sentiments):
    if not sentiments:
        return 'neutral'

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
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
print(aspect_sentiments_df[['reviewTitle', 'reviewBody', 'title_sentiment',
      'body_sentiment', 'title_aspects', 'title_sentiments', 'body_aspects', 'body_sentiments']])
