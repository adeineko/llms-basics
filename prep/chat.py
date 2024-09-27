
def choose_api():
    global choose_model, generate_answer, chosen_api, is_with_context
    is_with_context = False
    chosen_api = input(
        "Available APIs\n1) Goose AI\n2) HuggingFace\n3) Ollama (local)\n\nChoose an API to use: ")
    if chosen_api == "1":
        import apis.goose
        choose_model = apis.goose.choose_model
        generate_answer = apis.goose.generate_answer
    elif chosen_api == "2":
        import apis.huggingface
        choose_model = apis.huggingface.choose_model
        generate_answer = apis.huggingface.generate_answer
        is_with_context = True
    elif chosen_api == "3":
        import apis.ollama
        choose_model = apis.ollama.choose_model
        generate_answer = apis.ollama.generate_answer
    else:
        print("Invalid API. Please try again.")
        choose_api()


def ask_context():
    context = input(
        "\nDescribe what you're going to ask me about (or type exit)\n")
    if context == "exit":
        exit()
    return context


def ask_question():
    question = input("\nEnter your question (or type exit)\n")
    if question == "exit":
        exit()
    return question


def main():
    choose_api()
    model = choose_model()
    if is_with_context:
        context = ask_context()
    else:
        context = ""
    while True:
        question = ask_question()
        answer = generate_answer(model, question, context)
        print("\nGenerated answer:")
        print(answer)


if __name__ == "__main__":
    main()
