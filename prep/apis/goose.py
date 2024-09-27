import openai
import os
from dotenv import load_dotenv
from config.definitions import ROOT_DIR

# Load variables from .env file
load_dotenv(os.path.join(ROOT_DIR, '.env.local'))
openai.api_base = "https://api.goose.ai/v1"
openai.api_key = os.getenv("GOOSE_API_KEY")


def choose_model():
    print("Choose a model:")
    # List Engines (since Goose AI uses 'engines' endpoint)
    engines = openai.Engine.list()
    # Print all engine IDs
    for engine in engines['data']:
        print(engine['id'])
    default_model = engines['data'][0]['id']
    model = input(
        f"\nEnter the model you would like to use (default: {default_model})\n") or default_model
    return model


def generate_answer(model, question, context):
    # Create the prompt
    prompt = f"Question: {question}\nAnswer:"

    # Create a completion
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            n=1,
            stop=["\n", "Question:", "Answer:"]
        )
        # Extract the generated text
        answer = response.choices[0].text.strip()
        return answer
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"
