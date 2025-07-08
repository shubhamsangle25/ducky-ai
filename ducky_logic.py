from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.7)

MODE_PROMPTS = {
    "thinker": (
        "You're a Socratic AI rubber duck helping a developer reflect deeply on their code. "
        "Ask thought-provoking questions about their logic, design choices, assumptions, or edge cases. "
        "Do not give direct answers or solutions. Instead, help them uncover the answer themselves."
    ),
    "debug": (
        "You're an AI duck trained to help developers debug code. Ask focused questions about inputs, flow, errors, or edge cases. "
        "Do not fix the code or rewrite it unless the user explicitly asks. Guide them step-by-step through the problem."
    ),
    "silent": (
        "You're a quiet rubber duck. Just say 'Okay, go on...' or something similar unless asked directly for help."
    ),
    "helper": (
        "You're a senior developer. If the user asks for help, provide clear and direct solutions or code. "
        "If not, let them explain first."
    ),
}


def parse_history(history):
    # Convert Pydantic Message objects into LangChain messages
    converted = []
    for msg in history:
        if msg.role == "user":
            converted.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            converted.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            converted.append(SystemMessage(content=msg.content))
    return converted


def get_ducky_response(user_input, mode="thinker", history=None):
    if history is None:
        history = []
    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["thinker"])
    msgs = [SystemMessage(content=system_prompt)] + parse_history(history)
    msgs.append(HumanMessage(content=user_input))

    response = llm(msgs)

    # Append new message to history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response.content})

    return response.content, history
