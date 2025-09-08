import os
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool as langchain_tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the LLM (Groq LLM)
# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = init_chat_model("openai:gpt-4.1-nano")

# @langchain_tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     print(f"Requesting human assistance for query: {query}")
#     human_response = interrupt({"query": query})
#     print(f"Received human response: {human_response['data']}")
#     return human_response["data"]

@langchain_tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)

# Initialize Tavily Search tool (For web search)
tool = TavilySearch(max_results=2) # search results is 2
tools = [tool, human_assistance]

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str
    
def chatbot(state: State):
    # use llm_with_tools to access tools which llm has access to
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Initialize the StateGraph
graph_builder = StateGraph(State)


# The first argument is the unique node name. The second argument is the function or object that will be called whenever the node is used.
graph_builder.add_node("chatbot", chatbot)

# Add a ToolNode to manage the tools
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")

# Define the start and end of the graph
graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)
# graph = graph_builder.compile()

# Complile the graph with an in-memory checkpointer
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# Function to call the graph and stream updates
def stream_graph_updates(user_input: str, thread_id: str):
    '''
        Call the graph with user input and thread ID to maintain context. 
        The context is maintained in-memory with thread_id as the key.
    '''
    # event = graph.stream({"messages": [{"role": "user", "content": user_input}]})
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": thread_id}},
    )
    # stream_mode="values",

    for event in events:
        # Print messages directly if event type is values
        # event["messages"][-1].pretty_print()

        # If event is a dict with node names as keys
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def replay_state_history(graph, config):
    to_replay = None
    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)
        if len(state.values["messages"]) == 2:
            # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
            to_replay = state
        
    print(to_replay.next)
    print(to_replay.config)
    # The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

def test_human_in_loop():
    '''
        Test the human assistance tool in the graph.
    '''
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    replay_state_history(graph, config)


def test_chatbot():
    '''
        Test the chatbot with websearch tool with threads for memory
    '''
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, thread_id="1")
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input, thread_id="1")
            break


def custom_state():
    user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
    )
    config = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
    )

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)

    print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})
    graph.update_state(config, {"name": "LangGraph (library)"})

    snapshot = graph.get_state(config)

    print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})

if __name__ == "__main__":
    test_human_in_loop()
    # test_chatbot()
    # custom_state()