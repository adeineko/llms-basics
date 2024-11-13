import pandas as pd

# Load the data
file_path = '../sentiment-analysis/llm-based/analysis_results-v2.csv'
reviews_df = pd.read_csv(file_path)


def generate_response(row):
    # Check if aspect is not null
    if pd.notnull(row['aspect']):
        aspect_lower = row['aspect'].lower()
        if row['sentiment'] == 'positive':
            return f"Thank you for highlighting the {aspect_lower}. We're glad you're satisfied!"
        elif row['sentiment'] == 'negative':
            return f"We're sorry about the issues with {aspect_lower}. We'll work on improving this aspect!"
        else:  # neutral or other sentiments
            return f"Thank you for sharing your thoughts on the {aspect_lower}."
    else:
        # Generic response if aspect is null
        return "Thank you for your feedback!"


# Apply the response generation to each row
reviews_df['response'] = reviews_df.apply(generate_response, axis=1)

# Save the results to a new CSV file
output_file_path = 'reviews_with_responses.csv'
reviews_df.to_csv(output_file_path, index=False)

print(f"Responses saved to {output_file_path}")
