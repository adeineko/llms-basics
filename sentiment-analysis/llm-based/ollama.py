import csv
import requests
import json
import re

# Base URL of your Ollama Llama3 API
BASE_URL = 'http://localhost:11434/api/generate'


def query_llama3(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }
    # Enable streaming response
    response = requests.post(BASE_URL, json=payload,
                             headers=headers, stream=True)
    response.raise_for_status()

    response_text = ''
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            # Each line is a JSON object
            data = json.loads(decoded_line)
            if 'response' in data:
                # Concatenate the 'response' field
                response_text += data['response']

    return response_text.strip()


def analyze_review(review_body):
    prompt = f'''
Analyze the following product review for aspects and sentiments:

Review: "{review_body}"

Instructions:
- Identify all aspects mentioned in the review.
- Determine the sentiment (positive, negative, or neutral) for each aspect.
- Provide the results **only** in valid JSON format without any additional text or explanation.

Output Format:
{{
  "aspects": [
    {{"aspect": "aspect1", "sentiment": "positive"}},
    {{"aspect": "aspect2", "sentiment": "negative"}}
  ]
}}
'''
    response = query_llama3(prompt)
    # Attempt to extract JSON from the response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            analysis = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            print("Response Text:", response)
            analysis = {"error": "Invalid JSON", "raw_response": response}
    else:
        print("No JSON found in the response.")
        print("Response Text:", response)
        analysis = {"error": "No JSON found", "raw_response": response}
    return analysis


# Aspect categories
aspect_categories = {
    'price': ['price', 'cost', 'value', 'expensive', 'cheap'],
    'quality': ['quality', 'build', 'durability', 'materials', 'design'],
    'comfort': ['comfort', 'comfortable'],
    'functionality': ['functionality', 'performance', 'works', 'working', 'function', 'use'],
    'appearance': ['look', 'appearance', 'design', 'style', 'color'],
    'service': ['service', 'customer service', 'support', 'shipping', 'delivery'],
}


def categorize_aspect(aspect):
    for category, keywords in aspect_categories.items():
        if any(keyword in aspect.lower() for keyword in keywords):
            return category
    return 'other'  # Default category


def process_reviews(csv_file, output_csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f_in, open(output_csv_file, 'w', newline='',
                                                             encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ['reviewTitle', 'reviewBody',
                      'aspect', 'categorized_aspect', 'sentiment']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            review_title = row['reviewTitle']
            review_body = row['reviewBody']
            analysis = analyze_review(review_body)
            print(f"Review Title: {review_title}")
            print(f"Analysis: {analysis}")
            print("-" * 50)

            # Check if analysis contains 'aspects' key
            if 'aspects' in analysis:
                for aspect_info in analysis['aspects']:
                    aspect = aspect_info.get('aspect', '')
                    sentiment = aspect_info.get('sentiment', '')
                    # Categorize the aspect
                    categorized_aspect = categorize_aspect(aspect)
                    writer.writerow({
                        'reviewTitle': review_title,
                        'reviewBody': review_body,
                        'aspect': aspect,
                        'categorized_aspect': categorized_aspect,
                        'sentiment': sentiment
                    })
            else:
                # If analysis is invalid or has an error, write a row with empty aspect and sentiment
                writer.writerow({
                    'reviewTitle': review_title,
                    'reviewBody': review_body,
                    'aspect': '',
                    'categorized_aspect': '',
                    'sentiment': '',
                    # Optionally, include error info
                    # 'error': analysis.get('error', '')
                })


if __name__ == "__main__":
    csv_file = '../SentimentAssignmentReviewCorpus.csv'
    output_csv_file = 'analysis_results-v2.csv'
    process_reviews(csv_file, output_csv_file)
