from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Google GenAI (forward-ref fix for some stacks) ---
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_core.caches import BaseCache  # noqa: F401
    ChatGoogleGenerativeAI.model_rebuild()
except Exception:
    pass
# ------------------------------------------------------

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Add it as a Secret in your Hugging Face Space (Settings → Secrets)."
    )

system_prompt = """
You are Einstein.
Answer from Einstein’s point of view. Share personal anecdotes from life even if not asked.
Keep answers to 2–6 sentences and be a bit humorous.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ]
)

chain = prompt | llm | StrOutputParser()


def chat(user_input, hist):
    # Gradio (type="messages") → list[{"role": "...", "content": "..."}]
    langchain_history = []
    for item in hist or []:
        if item.get("role") == "user":
            langchain_history.append(HumanMessage(
                content=item.get("content", "")))
        elif item.get("role") == "assistant":
            langchain_history.append(
                AIMessage(content=item.get("content", "")))

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
body { background: radial-gradient(1200px 600px at 10% 10%, #f3f6ff 0%, #ffffff 60%) no-repeat; }
#app-container { max-width: 980px; margin: 0 auto; }
.header { background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #374151 100%); color: #fff; border-radius: 18px; padding: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
.header h1 { margin: 0; font-size: 28px; letter-spacing: 0.2px; }
.header p { margin: 6px 0 0; opacity: 0.85 }
.chatcard { border-radius: 18px !important; box-shadow: 0 10px 30px rgba(16,24,40,0.06); border: 1px solid #eef2f7; overflow: hidden; }
.gr-chatbot .message { padding: 10px 12px; border-radius: 14px !important; }
.gr-chatbot .message.user { background: #eef2ff !important; }
.gr-chatbot .message.bot { background: #f8fafc !important; }
.controls { display:flex; gap:10px; align-items:center; }
.footer { color:#6b7280; font-size: 13px; text-align:center; margin-top: 12px; }
"""

theme = gr.themes.Soft()

page = gr.Blocks(title="Chat with Einstein", theme=theme, css=custom_css)
with page:
    gr.HTML('<div id="app-container">')
    with gr.Column():
        with gr.Group(elem_classes="header"):
            gr.Markdown("# Chat with Einstein")
            gr.Markdown(
                "Have a witty conversation with *Professor Einstein*. "
                "Ask about physics, life, coffee ☕, or violin 🎻."
            )

        with gr.Group(elem_classes="chatcard"):
            # If the file "einstein.png" might not exist, set to None or use a public URL
            chatbot = gr.Chatbot(
                type="messages",
                # or ["https://.../einstein.png", None]
                avatar_images=[None, None],
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

        gr.HTML(
            '<div class="footer">Made with Python • LangChain • Gemini • Gradio</div>')
    gr.HTML("</div>")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    send_btn.click(chat, [msg, chatbot], [msg, chatbot])

    ex1.click(lambda: "What is time dilation?", None, msg)
    ex2.click(lambda: "Share a funny story from your life.", None, msg)
    ex3.click(lambda: "How did you think about relativity?", None, msg)

    clear.click(clear_chat, outputs=[msg, chatbot])

# Expose for Hugging Face Spaces auto-discovery
demo = page

if __name__ == "__main__":
    page.launch(share=True)  # local runs only
