import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import spacy
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load Aspect-Based Sentiment Analysis model
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
absa_tokenizer = AutoTokenizer.from_pretrained(absa_model_name)
absa_model = AutoModelForSequenceClassification.from_pretrained(
    absa_model_name)

df = pd.read_csv("../SentimentAssignmentReviewCorpus.csv")
df = df.dropna(subset=["reviewBody", "reviewTitle"],
               axis=0).astype(str)
reviews = df[['reviewTitle', 'reviewBody']]
final_results = []


def extract_aspects(sentence):
    """Extracts aspects from the sentence using SpaCy."""
    aspects = set()
    doc = nlp(sentence)
    for token in doc:
        # Identify nouns and proper nouns as potential aspects
        if token.pos_ in ["NOUN", "PROPN"]:
            aspects.add(token.text)
    return list(aspects)


def perform_absa(sentence, aspect):
    """Performs ABSA for a given sentence and aspect."""
    inputs = absa_tokenizer(
        f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()[0]


def calculate_overall_sentiment(sentiments):
    """Calculates overall sentiment based on aspect sentiments."""
    if sentiments:
        total_negative = sum(s['negative'] for s in sentiments)
        total_neutral = sum(s['neutral'] for s in sentiments)
        total_positive = sum(s['positive'] for s in sentiments)
        num_aspects = len(sentiments)

        avg_negative = total_negative / num_aspects
        avg_neutral = total_neutral / num_aspects
        avg_positive = total_positive / num_aspects

        if avg_positive > avg_negative and avg_positive > avg_neutral:
            return 'positive'
        elif avg_negative > avg_positive and avg_negative > avg_neutral:
            return 'negative'
        else:
            return 'neutral'
    return 'neutral'


for index, row in reviews.iterrows():
    title = row['reviewTitle']
    body = row['reviewBody']

    # Analyze title for aspects
    title_aspects = extract_aspects(title)
    title_sentiments = []
    for aspect in title_aspects:
        probs = perform_absa(title, aspect)
        title_sentiments.append({
            'aspect': aspect,
            'negative': probs[0],
            'neutral': probs[1],
            'positive': probs[2]
        })

    overall_sentiment_title = calculate_overall_sentiment(title_sentiments)

    # Analyze body for aspects
    body_aspects = extract_aspects(body)
    body_sentiments = []
    for aspect in body_aspects:
        probs = perform_absa(body, aspect)
        body_sentiments.append({
            'aspect': aspect,
            'negative': probs[0],
            'neutral': probs[1],
            'positive': probs[2]
        })

    overall_sentiment_body = calculate_overall_sentiment(body_sentiments)

    # Append results for the current review
    final_results.append({
        'reviewTitle': title,
        'reviewBody': body,
        'title_sentiment': overall_sentiment_title,
        'body_sentiment': overall_sentiment_body,
        'title_aspects': title_aspects,
        'title_sentiments': title_sentiments,
        'body_aspects': body_aspects,
        'body_sentiments': body_sentiments
    })

# Create a DataFrame to save the results
results_df = pd.DataFrame(final_results)
results_df.to_csv("absa_results.csv", index=False)
print("ABSA results extracted and saved to 'absa_results.csv'")
