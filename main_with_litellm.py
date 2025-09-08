import os
import json
from datetime import datetime
from typing import Annotated, TypedDict, Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
import requests

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["POCKETBASE_URL"] = os.getenv("POCKETBASE_URL", "http://localhost:8090")
os.environ["POCKETBASE_EMAIL"] = os.getenv("POCKETBASE_EMAIL")
os.environ["POCKETBASE_PASSWORD"] = os.getenv("POCKETBASE_PASSWORD")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class PocketBaseCheckpointSaver(BaseCheckpointSaver):
    """Custom checkpoint saver that stores conversation state in PocketBase."""
    
    def __init__(self, base_url: str, email: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.token = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with PocketBase and get admin token."""
        try:
            response = requests.post(
                f"{self.base_url}/api/admins/auth-with-password",
                json={
                    "identity": self.email,
                    "password": self.password
                }
            )
            if response.status_code == 200:
                self.token = response.json()["token"]
                print("✅ PocketBase authentication successful")
            else:
                print(f"❌ PocketBase authentication failed: {response.status_code}")
        except Exception as e:
            print(f"❌ PocketBase connection error: {e}")
    
    def _get_headers(self):
        """Get headers with authentication token."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def _ensure_collection_exists(self):
        """Ensure the conversations collection exists in PocketBase."""
        try:
            # Try to get the collection
            response = requests.get(
                f"{self.base_url}/api/collections/conversations",
                headers=self._get_headers()
            )
            
            if response.status_code == 404:
                # Create the collection if it doesn't exist
                collection_schema = {
                    "name": "conversations",
                    "type": "base",
                    "schema": [
                        {
                            "name": "thread_id",
                            "type": "text",
                            "required": True
                        },
                        {
                            "name": "checkpoint_id",
                            "type": "text",
                            "required": True
                        },
                        {
                            "name": "state",
                            "type": "json",
                            "required": True
                        },
                        {
                            "name": "metadata",
                            "type": "json",
                            "required": False
                        },
                        {
                            "name": "created_at",
                            "type": "date",
                            "required": True
                        }
                    ]
                }
                
                response = requests.post(
                    f"{self.base_url}/api/collections",
                    json=collection_schema,
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    print("✅ Created conversations collection in PocketBase")
                else:
                    print(f"❌ Failed to create collection: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error ensuring collection exists: {e}")
    
    def put(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata) -> None:
        """Save a checkpoint to PocketBase."""
        try:
            self._ensure_collection_exists()
            
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            checkpoint_id = checkpoint["id"]
            
            # Convert messages to serializable format
            serializable_state = {
                "messages": [
                    {
                        "type": msg.type if hasattr(msg, 'type') else 'unknown',
                        "content": msg.content if hasattr(msg, 'content') else str(msg),
                        "id": getattr(msg, 'id', None)
                    } for msg in checkpoint["channel_values"]["messages"]
                ]
            }
            
            record_data = {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "state": serializable_state,
                "metadata": metadata,
                "created_at": datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{self.base_url}/api/collections/conversations/records",
                json=record_data,
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                print(f"✅ Saved checkpoint {checkpoint_id} to PocketBase")
            else:
                print(f"❌ Failed to save checkpoint: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
    
    def get_tuple(self, config: Dict[str, Any]) -> Optional[tuple]:
        """Get the latest checkpoint for a thread."""
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            
            response = requests.get(
                f"{self.base_url}/api/collections/conversations/records",
                params={
                    "filter": f"thread_id = '{thread_id}'",
                    "sort": "-created_at",
                    "perPage": 1
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                records = response.json().get("items", [])
                if records:
                    record = records[0]
                    # Reconstruct checkpoint from stored data
                    checkpoint = {
                        "id": record["checkpoint_id"],
                        "channel_values": {
                            "messages": record["state"]["messages"]
                        }
                    }
                    metadata = record.get("metadata", {})
                    return (checkpoint, metadata)
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting checkpoint: {e}")
            return None
    
    def list(self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None, before: Optional[str] = None, limit: Optional[int] = None) -> List[tuple]:
        """List checkpoints for a thread."""
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            
            params = {
                "filter": f"thread_id = '{thread_id}'",
                "sort": "-created_at"
            }
            
            if limit:
                params["perPage"] = limit
            
            response = requests.get(
                f"{self.base_url}/api/collections/conversations/records",
                params=params,
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                records = response.json().get("items", [])
                checkpoints = []
                
                for record in records:
                    checkpoint = {
                        "id": record["checkpoint_id"],
                        "channel_values": {
                            "messages": record["state"]["messages"]
                        }
                    }
                    metadata = record.get("metadata", {})
                    checkpoints.append((checkpoint, metadata))
                
                return checkpoints
            
            return []
            
        except Exception as e:
            print(f"❌ Error listing checkpoints: {e}")
            return []
    
    def get_conversation_history(self, thread_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a specific thread using PocketBase API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/collections/conversations/records",
                params={
                    "filter": f"thread_id = '{thread_id}'",
                    "sort": "-created_at",
                    "perPage": limit
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                records = response.json().get("items", [])
                return records
            return []
            
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return []
    
    def delete_conversation(self, thread_id: str) -> bool:
        """Delete all records for a specific thread using PocketBase API."""
        try:
            # First get all records for the thread
            records = self.get_conversation_history(thread_id, limit=1000)
            
            for record in records:
                response = requests.delete(
                    f"{self.base_url}/api/collections/conversations/records/{record['id']}",
                    headers=self._get_headers()
                )
                if response.status_code != 204:
                    print(f"❌ Failed to delete record {record['id']}")
                    return False
            
            print(f"✅ Deleted conversation thread: {thread_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting conversation: {e}")
            return False
    
    def get_all_threads(self) -> List[Dict]:
        """Get all unique thread IDs using PocketBase API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/collections/conversations/records",
                params={
                    "fields": "thread_id,created_at",
                    "sort": "-created_at",
                    "perPage": 1000
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                records = response.json().get("items", [])
                # Group by thread_id and get latest record for each
                threads = {}
                for record in records:
                    thread_id = record["thread_id"]
                    if thread_id not in threads:
                        threads[thread_id] = {
                            "thread_id": thread_id,
                            "last_activity": record["created_at"],
                            "message_count": 1
                        }
                    else:
                        threads[thread_id]["message_count"] += 1
                
                return list(threads.values())
            return []
            
        except Exception as e:
            print(f"❌ Error getting threads: {e}")
            return []
    
    def search_conversations(self, query: str) -> List[Dict]:
        """Search conversations by content using PocketBase API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/collections/conversations/records",
                params={
                    "filter": f"state.messages.content ~ '{query}'",
                    "sort": "-created_at",
                    "perPage": 50
                },
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                records = response.json().get("items", [])
                return records
            return []
            
        except Exception as e:
            print(f"❌ Error searching conversations: {e}")
            return []

def tavily_search(query: str) -> str:
    """Search the web using Tavily API for real-time information."""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.getenv("TAVILY_API_KEY"),
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 3
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            answer = data.get("answer", "")
            
            if answer:
                return f"Answer: {answer}\n\nSources: {', '.join([r.get('title', '') for r in results[:3]])}"
            else:
                return f"Search results: {', '.join([r.get('title', '') + ': ' + r.get('content', '')[:200] for r in results[:3]])}"
        else:
            return f"Search failed with status {response.status_code}"
    except Exception as e:
        return f"Search error: {str(e)}"

def chatbot(state: State):
    # Get the last user message
    user_message = state["messages"][-1].content
    
    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "Search the web for real-time information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Use LiteLLM Proxy Server with tools
    response = requests.post(
        "http://localhost:4000/chat/completions",
        json={
            "model": "groq-llama",  # This matches the model_name in config
            "messages": [{"role": "user", "content": user_message}],
            "tools": tools,
            "tool_choice": "auto"
        }
    )
    
    # Return the response in the expected format
    from langchain_core.messages import AIMessage
    ai_message = response.json()["choices"][0]["message"]
    
    # Check if the LLM wants to use a tool
    if ai_message.get("tool_calls"):
        # Convert tool calls to the format LangChain expects
        from langchain_core.messages.tool import ToolCall
        import json
        tool_calls = []
        for tc in ai_message["tool_calls"]:
            # Parse arguments safely
            if isinstance(tc["function"]["arguments"], dict):
                args = tc["function"]["arguments"]
            else:
                args = json.loads(tc["function"]["arguments"])
            
            tool_calls.append(ToolCall(
                name=tc["function"]["name"],
                args=args,
                id=tc["id"]
            ))
        
        return {"messages": [AIMessage(
            content=ai_message.get("content", ""),
            tool_calls=tool_calls
        )]}
    else:
        return {"messages": [AIMessage(content=ai_message["content"])]}

def tool_executor(state: State):
    """Execute tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        from langchain_core.messages import ToolMessage
        tool_messages = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "tavily_search":
                result = tavily_search(tool_args["query"])
                tool_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
        
        return {"messages": tool_messages}
    
    return {"messages": []}

graph_builder = StateGraph(State)
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_executor)
graph_builder.add_edge(START, "chatbot")

def route_tools(state: State):
    """Route to tools if needed, otherwise end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message is an AI message with tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Check if the last message is a tool message (result from tool execution)
    elif hasattr(last_message, 'type') and last_message.type == 'tool':
        return "chatbot"  # Go back to chatbot to process tool result
    else:
        return END

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "chatbot": "chatbot", END: END}
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_conditional_edges(
    "tools",
    route_tools,
    {"tools": "tools", "chatbot": "chatbot", END: END}
)
# Initialize PocketBase checkpoint saver
try:
    pocketbase_url = os.getenv("POCKETBASE_URL", "http://localhost:8090")
    pocketbase_email = os.getenv("POCKETBASE_EMAIL")
    pocketbase_password = os.getenv("POCKETBASE_PASSWORD")
    
    if pocketbase_email and pocketbase_password:
        memory = PocketBaseCheckpointSaver(pocketbase_url, pocketbase_email, pocketbase_password)
        print("✅ Using PocketBase for conversation persistence")
    else:
        from langgraph.checkpoint.memory import InMemorySaver
        memory = InMemorySaver()
        print("⚠️  PocketBase credentials not found, using InMemorySaver")
except Exception as e:
    from langgraph.checkpoint.memory import InMemorySaver
    memory = InMemorySaver()
    print(f"⚠️  PocketBase initialization failed: {e}, using InMemorySaver")

graph = graph_builder.compile(checkpointer=memory)

try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
except ImportError:
    # IPython not available - graph visualization skipped
    print("Graph visualization skipped (IPython not installed)")
except Exception:
    # This requires some extra dependencies and is optional
    print("Graph visualization failed")

def stream_graph_updates(user_input: str, thread_id: str = "default"):
    from langchain_core.messages import HumanMessage
    
    # Create config with thread_id for conversation persistence
    config = {"configurable": {"thread_id": thread_id}}
    
    for event in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    ):
        for value in event.values():
            if value.get("messages"):
                last_message = value["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    print("Assistant:", last_message.content)

# PocketBase API helper functions
def show_conversation_history(thread_id: str):
    """Display conversation history using PocketBase API."""
    if hasattr(memory, 'get_conversation_history'):
        history = memory.get_conversation_history(thread_id, limit=5)
        if history:
            print(f"\n📜 Recent conversation history for {thread_id}:")
            for record in history[:3]:  # Show last 3 messages
                messages = record.get("state", {}).get("messages", [])
                for msg in messages[-2:]:  # Last 2 messages per checkpoint
                    role = "User" if msg.get("type") == "human" else "Assistant"
                    content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
                    print(f"  {role}: {content}")
        else:
            print(f"No history found for {thread_id}")

def list_all_conversations():
    """List all conversations using PocketBase API."""
    if hasattr(memory, 'get_all_threads'):
        threads = memory.get_all_threads()
        if threads:
            print("\n📋 All conversations:")
            for thread in threads[:10]:  # Show last 10
                print(f"  {thread['thread_id']} - {thread['message_count']} messages - {thread['last_activity'][:10]}")
        else:
            print("No conversations found")

def search_conversations(query: str):
    """Search conversations using PocketBase API."""
    if hasattr(memory, 'search_conversations'):
        results = memory.search_conversations(query)
        if results:
            print(f"\n🔍 Search results for '{query}':")
            for result in results[:5]:  # Show top 5 results
                thread_id = result.get("thread_id", "unknown")
                messages = result.get("state", {}).get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    content = last_msg.get("content", "")[:100] + "..." if len(last_msg.get("content", "")) > 100 else last_msg.get("content", "")
                    print(f"  {thread_id}: {content}")
        else:
            print(f"No results found for '{query}'")

# Conversation management
current_thread_id = "default"
print("🤖 LangGraph Chatbot with Tavily Search & PocketBase Persistence")
print("Commands:")
print("  'new' - new conversation")
print("  'history' - show conversation history")
print("  'list' - list all conversations")
print("  'search <query>' - search conversations")
print("  'delete' - delete current conversation")
print("  'quit' - exit")
print("-" * 60)

while True:
    try:
        user_input = input(f"\n[{current_thread_id}] User: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "new":
            current_thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"🆕 Started new conversation: {current_thread_id}")
            continue
        elif user_input.lower() == "history":
            show_conversation_history(current_thread_id)
            continue
        elif user_input.lower() == "list":
            list_all_conversations()
            continue
        elif user_input.lower().startswith("search "):
            query = user_input[7:].strip()
            if query:
                search_conversations(query)
            else:
                print("Please provide a search query: search <query>")
            continue
        elif user_input.lower() == "delete":
            if hasattr(memory, 'delete_conversation'):
                if memory.delete_conversation(current_thread_id):
                    current_thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    print(f"🗑️ Deleted conversation, started new: {current_thread_id}")
                else:
                    print("❌ Failed to delete conversation")
            else:
                print("❌ Delete not available with InMemorySaver")
            continue
        elif user_input.strip() == "":
            continue

        stream_graph_updates(user_input, current_thread_id)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input, current_thread_id)
        break