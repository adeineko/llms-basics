import openai
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv("../.env.local")


def choose_api():
    global api_key, api_url
    index = int(input("Choose an API (1-6): ") or 1)
    api_key = os.getenv("API_KEY_" + str(index))
    api_url = os.getenv("API_URL_" + str(index))
    # print(f"Using API {index} at {api_url} with key {api_key}")
    openai.api_key = api_key
    openai.api_base = api_url


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
    openai.api_key = api_key
    openai.api_base = api_url
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


def main():
    choose_api()
    model = choose_model()
    question = ask_question()
    answer = generate_answer(model, question)
    print("\nGenerated answer:")
    print(answer)


if __name__ == "__main__":
    main()
