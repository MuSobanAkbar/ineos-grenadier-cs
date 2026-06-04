import os 
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq()


#the temp will stay 0.0-0.2 for accuracy 
MODEL_NAME = "openai/gpt/120b"
TEMP=0.3

MAX_TOKEN_LIMIT = 1024

messages = [
    {
        "role":"system", "content":"all you say is testing"
    }
]

while True:
    user_input = input("You: ")
    messages.append({"role":"user", "content": user_input})
    try:
        message = client.chat.completions.create(
            model = MODEL_NAME,
            max_completion_tokens = MAX_TOKEN_LIMIT,
            messages = messages,
            temperature = TEMP,
             
             
        )
        if len(messages)>15:
            messages = [messages[0]] + messages [-14:]
        
        ai_response = message.choices[0].message.content
        print(f"AI: {ai_response}")
        messages.append({"role": "assistant", "content":ai_response})
    except Exception as e:
        print("Error has occured.")
