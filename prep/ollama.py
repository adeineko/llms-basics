from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import faiss # Facebook AI Similarity Search
import requests
import json
import wikipedia
import nltk # Natural Language Toolkit

nltk.download('punkt')
nltk.download('punkt_tab')

# Base URL of your Ollama Docker container with the llama3 model
BASE_URL = 'http://localhost:11434/api/generate'

# Set the language to English
wikipedia.set_lang("en")

# Fetch wikipedia page content
try:
    # Fetch the page content
    page = wikipedia.page("Hurricane Milton")
    content = page.content
except wikipedia.exceptions.PageError:
    print("The page 'Hurricane Milton' does not exist.")
    content = ""


# Step 2: Preprocess the content
sentences = sent_tokenize(content)
chunk_size = 5  # every 5 sentences grouped
chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

# Step 3: Generate embeddings for the chunks
# Each text chunk is converted into a numerical representation (embedding) using Sentence Transformers
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedding_model.encode(chunks).astype('float32')

# Step 4: Build a FAISS index which allows  efficient similarity searches
# This index enables quick retrieval of text chunks that are most relevant to a user's query.
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)


# Step 5: the retrieval function
def retrieve_relevant_documents(query, k=2):
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    return relevant_chunks


def query_llama3_model_stream(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(BASE_URL, json=payload, headers=headers, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")
        return

    print("Llama3: ", end="", flush=True)
    buffer = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            buffer += decoded_line
            try:
                data = json.loads(buffer)
                buffer = ""
                if 'response' in data:
                    print(data['response'], end="", flush=True)
            except json.JSONDecodeError:
                continue
    print()


def chat_with_llama3():
    print("Welcome to the Ollama Llama3 Chatbot with RAG!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_documents(user_input)
        context = "\n\n".join(relevant_docs)

        # Augment the prompt with the context
        augmented_prompt = f"""
You are an assistant knowledgeable about hurricanes.

Context:
{context}

Based on the above context, please answer the following question:

Question: {user_input}
Answer:
"""

        # Query the Llama3 model with the augmented prompt
        query_llama3_model_stream(augmented_prompt)


if __name__ == "__main__":
    chat_with_llama3()
