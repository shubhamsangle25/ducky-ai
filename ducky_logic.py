from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.7)

MODE_PROMPTS = {
    "rubber_duck": (
        "You're a helpful AI rubber duck. Help the developer reflect on their logic and debug their code. "
        "Ask thoughtful questions about their design, assumptions, and flow. "
        "Don't give direct answers unless they ask. Guide them to discover issues or improve their thinking."
    ),
    "helper": (
        "You're a senior developer. If the user asks for help, give direct and accurate solutions. "
        "If not, let them explain and ask specific questions first."
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
