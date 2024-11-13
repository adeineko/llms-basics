# AI Practice Hub

This repository contains solutions and implementations for exercises related to Large Language Models (LLMs) and sentiment analysis. The project demonstrates API usage, response generation, and multiple sentiment analysis techniques, organized for ease of use and extendibility.

---

## Project Structure

### **`configs/`**
- **`definitions.py`**: Configuration details and constants.

### **`apis/`**
- **`goose.py`**: Interface for the Goose API.
- **`huggingface.py`**: Integration with Hugging Face models.
- **`chat.py`**: Helper script for chat-based APIs.
- **`ollama.py`**: Ollama chatbot enhanced with RAG.

### **`response-generation/`**
- **`generate_response.py`**: Script for generating automated responses to reviews based on sentiments.
- **`reviews_with_responses.csv`**: Output data containing generated responses.

### **`sentiment-analysis/`**
- **`llm-based/`**: LLM-based sentiment analysis implementations.
- **`machine_learning/`**: Sentiment analysis using machine learning techniques.
- **`spacy-lexicon/`**: Lexicon-enhanced Spacy aspect-based analysis.
- **`SentimentAssignmentReviewCorpus.csv`**: Dataset for sentiment analysis tasks.

### Additional Files:
- **`docker-compose.yml`**: Pulls Ollama image to run chatbot.

---

## Getting Started

### Prerequisites
1. Python 3.10 or above.
2. Docker (if running `ollama.py`).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
