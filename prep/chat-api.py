import openai
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv(".env.local")

api_key = os.getenv("AI1_API_KEY")
api_base = "https://api.goose.ai/v1"

# Set the OpenAI API key and base URL
openai.api_key = api_key
openai.api_base = api_base


def choose_model():
    print("Choose a model:")
    # List Engines (since Goose AI uses 'engines' endpoint)
    engines = openai.Engine.list()
    # Print all engine IDs
    for engine in engines['data']:
        print(engine['id'])
    model = input(
        "Enter the model you would like to use (default: gpt-j-6b)\n") or "gpt-j-6b"
    print(f"Using model: {model}")
    return model


def ask_question():
    question = input("Please enter your question\n")
    return question


def generate_answer(model, question):
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
            stop=["\n", "Question:", "Answer:"],
            stream=False
        )
        # Extract the generated text
        answer = response.choices[0].text.strip()
        return answer
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"


def main():
    # TODO: choose API
    model = choose_model()
    question = ask_question()
    answer = generate_answer(model, question)
    print("\nGenerated answer:")
    print(answer)


if __name__ == "__main__":
    main()
