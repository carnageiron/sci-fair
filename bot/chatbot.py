'''import openai

# Set up the OpenAI API client
openai.api_key = "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"

# Set up the model and prompt
model_engine = "text-davinci-003"
prompt = "Hello, how are you today?"

# Generate a response
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

response = completion.choices[0].text
print(response)
'''

import openai
import gradio as gr

openai.api_key = "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"

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)
