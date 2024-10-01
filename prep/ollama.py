import requests
import json

# Base URL of your Ollama Docker container with the llama3 model
BASE_URL = 'http://localhost:11434/api/generate'


def query_llama3_model_stream(prompt):
    # Define the JSON payload
    payload = {
        "model": "llama3",
        "prompt": prompt
    }

    # Headers specifying the content type as JSON
    headers = {
        "Content-Type": "application/json"
    }

    # Send the POST request to the Llama3 API and stream the response
    response = requests.post(BASE_URL, json=payload, headers=headers, stream=True)

    if response.status_code == 200:
        # Iterate over the lines of the response to collect and print the `response` fields in real-time
        print("Llama3: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                try:
                    # Parse each line as JSON
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        # Print each part of the response as it's received
                        print(data['response'], end="", flush=True)
                except ValueError:
                    continue
        print()  # To ensure the next prompt appears on a new line
    else:
        # Print the error message
        print(f"Error: {response.status_code} - {response.text}")


def chat_with_llama3():
    print("Welcome to the Ollama Llama3 Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Exit the chat loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Query the Llama3 model with the user's input in streaming mode
        query_llama3_model_stream(user_input)


if __name__ == "__main__":
    chat_with_llama3()
