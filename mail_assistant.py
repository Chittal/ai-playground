import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_litellm import ChatLiteLLM
from langchain_core.tools import tool as langchain_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from pocketbase import PocketBase

from dotenv import load_dotenv
load_dotenv()

pb = PocketBase(os.getenv("POCKETBASE_URL"))
pb.admins.auth_with_password(os.getenv("POCKETBASE_EMAIL"), os.getenv("POCKETBASE_PASSWORD"))


class EmailState(TypedDict):
    email_id: str
    email_content: dict
    category: str
    status: str
    messages: Annotated[list, add_messages]

@langchain_tool
def email_reply(email_content: dict) -> str:
    """
    Use this tool ONLY when user explicitly asks to write or draft an email reply.
    Do NOT use this for general conversation, greetings, or questions.
    
    Args:
        email_content: Dictionary containing the original email content to reply to
    
    Returns:
        A drafted reply as plain text
    """
    prompt = f"""
        You are an assistant the drafts reply to email.
        Here is the email:
        {email_content}
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
    # state["status"] = "drafted_reply"
    # state["messages"].append(response)
    print("================================================================")
    print("EMAIL CONTENT", email_content)
    print("================================================================")
    return response.content


@langchain_tool
def add_to_meeting_or_task_to_calendar_tool():
    """
    ONLY use this tool when the user explicitly requests to schedule, add, or create calendar events.
    Do NOT use this for general conversation, greetings, or questions.
    Only use when user says: "schedule a meeting", "add to calendar", "create an event", "set up a meeting".
    
    Args:
        event_details: Description of the meeting/task to be added
    
    Returns:
        Confirmation that the event was added
    """
    print("================================================================")
    print("ADD TO MEETING OR TASK TO CALENDAR TOOL")
    print("================================================================")
    return "Added meeting or task to calendars"


llm = ChatLiteLLM(model="groq/llama-3.1-8b-instant")
llm = llm.bind_tools([email_reply, add_to_meeting_or_task_to_calendar_tool])

def chatbot(state: EmailState):
    # use llm_with_tools to access tools which llm has access to
    user_message = input("User: ")
    
    # Add system prompt to control behavior
    system_prompt = f"""You are an email assistant. You have access to these tools:
    - email_reply: ONLY use when user explicitly asks to draft/write/reply to an email
    - add_to_meeting_or_task_to_calendar_tool: ONLY use when user asks to schedule/calendar events
    
    For general conversation, greetings, or questions, respond normally without using tools.
    Only use tools when the user specifically requests email replies or calendar scheduling.
    Be helpful and conversational for other topics.
    
    CURRENT EMAIL BEING PROCESSED: {state["email_content"]}
    
    IMPORTANT: When calling email_reply, pass the current_email_content from the current email being processed."""
    
    # Prepare messages with system prompt
    messages = [{"role": "system", "content": system_prompt}] + state["messages"] + [{"role": "user", "content": user_message}]
    
    response = llm.invoke(messages)
    print("================================================================")
    print("LLM RESPONSE", response)
    print("================================================================")
    return {"messages": [response]}


def email_category(state: EmailState) -> EmailState:
    prompt = f"""
        You are an email classification assistant. Your ONLY job is to classify emails.
        
        IMPORTANT: Do NOT use any tools. Just classify the email and return the category.
        
        Classify this email into ONE of these categories:

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
        Do NOT call any tools - just return the category.
    """
    print(prompt)
    print("invoking llm")
    
    # Use the same LLM with tools, but with strong prompt to not use them
    response = llm.invoke(prompt)
    
    print("================================================================")
    print("CATEGORY ASSIGNED", response.content)
    print(response)
    print("================================================================")
    if response.content:
        state["category"] = response.content
        state["status"] = "category_assigned"
        email_data = {
            "email_content": state["email_content"],
            "email_category": response.content
        }
        print("EMAIL DATA", email_data)

        new_email = pb.collection("emails").create(email_data)
        print(f"Created email: {new_email}")
    else:
        state["status"] = "category_not_assigned"
    return state


def should_continue(state: EmailState) -> str:
    last_message = state["messages"][-1]
    # Check for quit words
    if hasattr(last_message, 'content'):
        content = last_message.content.lower()
        print("CONTENT", content)
        if any(word in content for word in ["quit", "exit", "bye", "stop"]):
            print("QUIT WORDS FOUND")
            return "end"
    # Check if there are actual tool calls
    print("tool calls check")
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("TOOL CALLS FOUND")
        return "tools"
    # Otherwise, continue conversation
    return "chatbot"  # or "chatbot" if you want to loop


graph = StateGraph(EmailState)

tool_node = ToolNode(tools=[email_reply, add_to_meeting_or_task_to_calendar_tool])
graph.add_node("tools", tool_node)

graph.add_node("email_category", email_category)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "email_category")
graph.add_edge("email_category", "chatbot")
graph.add_conditional_edges(
    "chatbot",
    should_continue,
    {"end": END, "tools": "tools", "chatbot": "chatbot"}
)
graph.add_edge("tools", "chatbot")
# graph.add_edge("chatbot", END)

memory = InMemorySaver()
graph = graph.compile(checkpointer=memory)

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
    }, {"configurable": {"thread_id": 1}})
print(result)