# HuggingFace models using Inference API
import requests
import os
from dotenv import load_dotenv
from config.definitions import ROOT_DIR

# Load variables from .env file
load_dotenv(os.path.join(ROOT_DIR, '.env.local'))
get_models_url = "https://huggingface.co/api/models"
api_url = "https://api-inference.huggingface.co/models/"
api_key = os.getenv("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {api_key}"}


def choose_model(top_n=10):
    # Parameters to sort models by number of downloads
    params = {
        "sort": "downloads",  # Sort by downloads
        "direction": -1,      # Descending order
        "limit": top_n,        # Number of models to retrieve
        "filter": "question-answering"  # Filter by pipeline tag
    }

    try:
        response = requests.get(get_models_url, params=params)
        response.raise_for_status()
        models = response.json()
        print(f"Top {top_n} Hugging Face Models by Downloads:\n")
        for i, model in enumerate(models, 1):
            model_id = model.get("modelId", "Unknown Model")
            downloads = model.get("downloads", "N/A")
            print(f"{model_id} - Downloads: {downloads}")
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}\nResponse: {response.text}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    default_model = models[0].get("modelId", "deepset/roberta-base-squad2")
    model = input(
        f"\nEnter the model id you would like to use (default: {default_model})\n") or default_model
    return model


def generate_answer(model, question, context):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    try:
        response = requests.post(

            api_url + model, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("generated_text", data)['answer']
        return answer
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}\nResponse: {response.text}"
    except requests.exceptions.RequestException as err:
        return f"Error: {err}"
    except KeyError:
        return f"Unexpected response format: {response.json()}"
