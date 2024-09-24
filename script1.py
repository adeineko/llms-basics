import openai
import os
from dotenv import load_dotenv, dotenv_values

# loading variables from .env file
load_dotenv(".env.local")

openai.api_key = os.getenv("AI1_API_KEY")
openai.api_base = "https://api.goose.ai/v1"

# List Engines (Models)
engines = openai.Engine.list()
# Print all engines IDs
for engine in engines.data:
    print(engine.id)

# Create a completion, return results streaming as they are generated. Run with `python3 -u` to ensure unbuffered output.
completion = openai.Completion.create(
    engine="gpt-j-6b",
    prompt="Once upon a time there was a Goose. ",
    max_tokens=160,
    stream=True)

# Print each token as it is returned
for c in completion:
    print(c.choices[0].text, end='')

print("")
