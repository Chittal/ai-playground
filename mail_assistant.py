from typing import Annotated, List

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_litellm import ChatLiteLLM


class EmailState(TypedDict):
    email_id: str
    email_content: dict
    category: str
    status: str


def email_category(state: EmailState) -> EmailState:
    prompt = f"""
        You are an assistant that classifies emails into ONE of these categories:

        1. FAMILY
        2. FRIENDS
        3. SUBSCRIPTION
        4. SHOPPING
        5. JUNK
        6. TODO
        7. REPLY_REQUIRED
        8. UPCOMING_EVENT

        Email content:
        {state["email_content"]}

        Return exactly ONE category in UPPERCASE.  
        Do not add any explanation, numbering, or extra text.  
        Output must be only the category word.
    """
    print(prompt)
    print("invoking llm")
    llm = ChatLiteLLM(model="groq/llama-3.1-8b-instant")
    response = llm.invoke(prompt)
    print("llm response")
    print(response)
    state["category"] = response.content
    state["status"] = "category_assigned"
    return state


def email_reply(state: EmailState) -> EmailState:
    prompt = f"""
        You are an assistant the drafts reply to email.
        Here is the email:
        {state["email_content"]}
        Return the reply to the email as TEXT.
        Do not add any explanation, numbering, or extra text.
        Return only the reply as TEXT.
    """
    print(prompt)
    print("invoking llm")
    llm = ChatLiteLLM(model="groq/llama-3.1-8b-instant")
    response = llm.invoke(prompt)
    print("llm response")
    print(response)
    # save reply to pocketbase db
    state["status"] = "drafted_reply"
    return state


def add_to_meeting_or_task_to_calendar_tool():
    pass

def send_email_tool():
    pass

def user_action():
    pass

graph = StateGraph(EmailState)

graph.add_node("email_category", email_category)
graph.add_edge(START, "email_category")
graph.add_edge("email_category", END)
graph = graph.compile()

print("Invoking graph")
result = graph.invoke({
    "email_id": "1", 
    "email_content": {
        "From": { 
            "Email": "george.thomas@company.com", 
            "Name": "George Thomas" 
            },
            "Headers": { "X-IP": "210.34.12.45" },
            "Subject": "Updated Timeline Request",
            "Text": "Could you send me the updated project timeline?"
            }
    })
print(result)