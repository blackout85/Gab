from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
import torch

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", return_dict=True)
config = AutoConfig.from_pretrained("microsoft/DialoGPT-small", max_length=1024)

# Initialize the chatbot pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, config=config)

app = Flask(__name__)
chat_history = []

# Add a pre-prompt to make the chatbot behave like a conversational agent
pre_prompt = "The following is a conversation with an AI assistant. The assistant will help you with your questions and concerns. The assistant is helpful, inquisitive, and sassy. "

def chatbot_response(user_input):
    if len(chat_history) > 0:
        chat_history[-1]["bot"] = user_input

    new_entry = {"user": user_input, "bot": ""}
    chat_history.append(new_entry)

    response = chatbot(pre_prompt + user_input, max_length=1000, do_sample=True, temperature=2.5, top_p=0.4, top_k=50)[0]["generated_text"].replace(pre_prompt, "").strip()

    if len(chat_history) > 1:
        if response == chat_history[-2]["bot"]:
            response = chatbot(pre_prompt + user_input + " again", max_length=1000, do_sample=True, temperature=2.5, top_p=0.4, top_k=50)[0]["generated_text"].replace(pre_prompt, "").strip()

    chat_history[-1]["bot"] = response

    return response

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form["user_input"]
        bot_response = chatbot_response(user_input)

        # Limit chat history to only display the most recent 3 entries
        if len(chat_history) > 3:
            chat_history.pop(0)

        return render_template("chat_3.html", chat_history=chat_history, bot_response=bot_response)
    else:
        return render_template("chat_3.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run()
