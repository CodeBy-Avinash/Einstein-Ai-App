from dotenv import load_dotenv
import os
import gradio as gr


from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Google GenAI (fix pydantic forward refs on some stacks) ---
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    # Some langchain/google-genai combos need this to resolve forward refs
    from langchain_core.caches import BaseCache  # noqa: F401
    ChatGoogleGenerativeAI.model_rebuild()
except Exception:
    pass
# ---------------------------------------------------------------

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# Nice guard: fail early with a clear message if key is missing
if not gemini_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Set it in a .env file locally or as a Secret on your Space."
    )

system_prompt = """" 
You are Einstein.
Answer questions from your point of view. You will share personal things from life
even when the user dont't ask for it. For example, if the user asks about the theory of
relativity, you will share your personal experiences with it and not only explain the theory.
give in 2-6 sentences.
You should have a sense of humour.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (MessagesPlaceholder(variable_name="history")),
        ("user", "{input}"),
    ]
)

chain = prompt | llm | StrOutputParser()

print("Hi Albert, how can i help you today?")


def chat(user_input, hist):
    # Convert Gradio messages -> LangChain messages
    langchain_history = []
    for item in hist:
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            langchain_history.append(AIMessage(content=item["content"]))

    response = chain.invoke(
        {"input": user_input, "history": langchain_history})

    return "", hist + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
    ]


def clear_chat():
    return "", []


# -------------------- UI --------------------
custom_css = """
/* Page background */
body { background: radial-gradient(1200px 600px at 10% 10%, #f3f6ff 0%, #ffffff 60%) no-repeat; }

/* Center and constrain content */
#app-container { max-width: 980px; margin: 0 auto; }

/* Header card */
.header {
  background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #374151 100%);
  color: #fff; border-radius: 18px; padding: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}
.header h1 { margin: 0; font-size: 28px; letter-spacing: 0.2px; }
.header p { margin: 6px 0 0; opacity: 0.85 }

/* Chatbot card */
.chatcard {
  border-radius: 18px !important;
  box-shadow: 0 10px 30px rgba(16,24,40,0.06);
  border: 1px solid #eef2f7;
  overflow: hidden;
}

/* Chat bubbles spacing */
.gr-chatbot .message { padding: 10px 12px; border-radius: 14px !important; }
.gr-chatbot .message.user { background: #eef2ff !important; }
.gr-chatbot .message.bot { background: #f8fafc !important; }

/* Inputs row */
.controls { display:flex; gap:10px; align-items:center; }

/* Footer */
.footer {
  color:#6b7280; font-size: 13px; text-align:center; margin-top: 12px;
}
"""

theme = gr.themes.Soft()

page = gr.Blocks(title="Chat with Einstein", theme=theme, css=custom_css)

with page:
    gr.HTML('<div id="app-container">')
    with gr.Column():
        # Header
        with gr.Group(elem_classes="header"):
            gr.Markdown("# Chat with Einstein")
            gr.Markdown(
                "Have a witty conversation with *Professor Einstein*. "
                "Ask about physics, life, coffee ☕, or violin 🎻."
            )

        # Chat + input
        with gr.Group(elem_classes="chatcard"):
            chatbot = gr.Chatbot(
                type="messages",
                avatar_images=[None, "einstein.png"],
                show_label=False,
                height=480,
            )

            with gr.Row(elem_classes="controls"):
                msg = gr.Textbox(
                    placeholder="Ask Einstein anything… (e.g., Explain relativity like I'm 10)",
                    scale=9,
                    autofocus=True,
                    lines=1,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                ex1 = gr.Button("What is time dilation?")
                ex2 = gr.Button("Share a funny story from your life.")
                ex3 = gr.Button("How did you think about relativity?")
                clear = gr.Button("Clear Chat", variant="secondary")

        # Footer
        gr.HTML(
            '<div class="footer">Made with Python • LangChain • Gemini • Gradio</div>')

    gr.HTML("</div>")  # close app-container

    # Wire interactions
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    send_btn.click(chat, [msg, chatbot], [msg, chatbot])

    # Example buttons must return a STRING (not a tuple)
    ex1.click(lambda: "What is time dilation?", None, msg)
    ex2.click(lambda: "Share a funny story from your life.", None, msg)
    ex3.click(lambda: "How did you think about relativity?", None, msg)

    clear.click(clear_chat, outputs=[msg, chatbot])

# Local runs only; Spaces ignores __main__ block
if __name__ == "__main__":
    page.launch(share=True, favicon_path="einstein.png")
