import os
from uuid import uuid4
import operator
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_core.messages import ( AIMessage, HumanMessage, SystemMessage,)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import ( MessagesState, StateGraph, START, END,)
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import NotRequired
from langgraph.graph import MessagesState
import requests

load_dotenv(find_dotenv())

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY was not found. Add it to your .env file.")

if "TAVILY_API_KEY" not in os.environ:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("TAVILY_API_KEY loaded:", bool(os.getenv("TAVILY_API_KEY")))
print("LANGSMITH_API_KEY loaded:", bool(os.getenv("LANGSMITH_API_KEY")))

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "exercise-3-multi-node-langgraph"

model = ChatOpenAI(model="gpt-4o-mini",
                    temperature=0)

class State(MessagesState):
    route: NotRequired[Literal["calculator", "search_web", "database_lookup"]]
    context: NotRequired[list[str]]


class Route(TypedDict):
    """Decision on which specialized node should handle the request."""
    route: Literal[ "calculator", "search_web", "database_lookup"]

ROUTER_SYSTEM_MESSAGE = """You are a router.

Decide which specialized node should handle the user's request.

Use:
- calculator for arithmetic calculations
- search_web for information that requires web search
- database_lookup for requests about the user's ToDo list

Call the Route tool with the appropriate route."""


def router(state: State):
    """Choose the appropriate specialized node."""
    response = model.bind_tools([Route], parallel_tool_calls=False,).invoke([SystemMessage(
	content=ROUTER_SYSTEM_MESSAGE)] + state["messages"])

    return {"route": response.tool_calls[0]["args"]["route"]}


def route_request(state: State) -> Literal["calculator", "search_web", "database_lookup"]:
    """Route the request to the selected specialized node."""
    return state["route"]


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide the first integer by the second integer."""
    return a / b

tools = [add, multiply, divide]

llm_with_tools = model.bind_tools(tools)

calculator_system_message = SystemMessage(
    content=("You are a helpful assistant tasked with performing "
             "arithmetic on a set of inputs."))


def calculator(state: State):
    """Perform arithmetic using the calculator tools."""
    return {"messages": [llm_with_tools.invoke(
                [calculator_system_message] + state["messages"])]}


class SearchQuery(BaseModel):
    search_query: str = Field(description="Search query for retrieval.")

search_instructions = SystemMessage(content=
"""You will be given a conversation.

Your goal is to generate a well-structured query for use in web search.

Analyze the full conversation and pay particular attention to the
latest user question.

Convert the latest question into a concise Tavily search query.""")


def search_web(state: State) -> dict:
    """Retrieve documents from Tavily search."""

    structured_llm = model.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [search_instructions] + state["messages"])

    try:
        import json
        import subprocess

        payload = json.dumps({
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": search_query.search_query,
            "max_results": 3,
            "search_depth": "advanced",
        })

        curl_result = subprocess.run(
            ["curl", "-s", "https://api.tavily.com/search",
             "-X", "POST",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        if curl_result.returncode != 0:
            return {"context": [f"Tavily search failed: curl error: {curl_result.stderr}"]}

        search_docs = json.loads(curl_result.stdout).get("results", [])
    except Exception as error:
        return {"context": [f"Tavily search failed: {error}"]}

    if not search_docs:
        return {"context": ["No useful Tavily search results were found."]}

    formatted_search_docs = "\n\n---\n\n".join((
            f'<Document href="{doc.get("url", "")}">\n'
            f'Title: {doc.get("title", "")}\n'
            f'{doc.get("content", "")}\n'
            f"</Document>") for doc in search_docs)
    return {"context": [formatted_search_docs]}

def answer_search(state: State) -> dict:
    """Generate an answer using the retrieved web context."""
    context = "\n\n".join(state.get("context", []))
    system_message = SystemMessage(content=f"""You are a helpful assistant.

Answer the user's question using the following Tavily results.

If the search failed or produced no useful results, explain that
clearly instead of inventing information.

Search results:
{context}""")

    answer = model.invoke([system_message] + state["messages"])
    return {"messages": [answer]}


class TodoRequest(BaseModel):
    """Structured operation requested by the user."""

    action: Literal["list", "add", "update", "delete"] = Field(
        description=("The requested ToDo operation. Use 'list' when the user "
                     "only wants to view or ask questions about existing ToDos."))

    target: str | None = Field(default=None,
                               description=("The ID or current task name of the ToDo"
                                            "that should be updated or deleted."))

    target_position: int | None = Field(default=None,
                                        description=("If the user refers to the ToDo by its "
                                                     "position in the list (e.g. 'the third one', "
                                                     "'number 17', 'the hundred and third task'), "
                                                     "put the 1-based integer position here. "
                                                     "Leave unset if they refer to it by name or ID."))

    task: str | None = Field(default=None,
                             description=("The task description for a new ToDo, or the new task "
                                          "description when renaming an existing ToDo."))

    deadline: str | None = Field(default=None,
                                 description=("The new deadline. Leave this unset when"
                                              "the user did not request a deadline change."))

    status: str | None = Field(default=None,
                               description=("The new status, such as 'not started', "
                                            "'in progress', or 'completed'."))


TODO_SYSTEM_MESSAGE = """You manage the user's ToDo list.

Determine whether the user wants to:

- list or ask about existing ToDos
- add a new ToDo
- update an existing ToDo
- delete an existing ToDo

Rules:

1. Use "list" for questions that do not modify the ToDo list.
2. For "add", place the new task description in `task`.
3. For "update" or "delete", place the existing task name or ID in `target`.
4. For "update", include only the fields the user wants changed.
5. Do not invent task names, deadlines, statuses, or IDs.
6. Use the conversation history to understand references such as
   "it", "that task", or "the first one".
"""


def format_todos(todo_memories) -> str:
    """Create a readable numbered representation of stored ToDos."""

    if not todo_memories:
        return "The ToDo list is empty."

    lines = []

    for number, memory in enumerate(todo_memories, start=1):
        value = memory.value

        lines.append(f"{number}. ID: {memory.key}\n"
                     f"   Task: {value.get('task', 'Untitled')}\n"
                     f"   Deadline: {value.get('deadline') or 'No deadline'}\n"
                     f"   Status: {value.get('status', 'not started')}")

    return "\n".join(lines)


def find_todo(todo_memories, target: str | None, target_position: int | None = None):
    """Find one ToDo by its resolved position, ID, or task description."""

    if target_position is not None:
        if 1 <= target_position <= len(todo_memories):
            return todo_memories[target_position - 1], None
        return None, f"ToDo number {target_position} does not exist."

    if not target:
        return None, "Please specify which ToDo you mean."

    cleaned_target = target.strip().lower()

    exact_matches = [memory
                     for memory in todo_memories
                     if (memory.key.lower() == cleaned_target
                         or str(memory.value.get("task", "")).strip().lower() == cleaned_target)]

    if len(exact_matches) == 1:
        return exact_matches[0], None

    partial_matches = [memory
                       for memory in todo_memories
                       if cleaned_target
                       in str(memory.value.get("task", "")).strip().lower()]

    if len(partial_matches) == 1:
        return partial_matches[0], None

    if len(partial_matches) > 1:
        matching_tasks = "\n".join(f"- {memory.value.get('task', 'Untitled')}"
                                   for memory in partial_matches)

        return None, ("More than one ToDo matches that description. "
                      "Please be more specific:\n"
                      f"{matching_tasks}")
    return None, f'I could not find a ToDo matching "{target}".'


def database_lookup(state: State,
                    config: RunnableConfig,
                    store: BaseStore) -> dict:
    """List, add, update, or delete the user's ToDos."""

    user_id = (config.get("configurable", {}).get("user_id") or "Test")
    todo_namespace = ("todo", user_id)
    todo_memories = list(store.search(todo_namespace))
    if not todo_memories:
        store.put(todo_namespace,
                  "travel_todo",
                  {"task": "Finish booking travel to Hong Kong",
                   "deadline": "End of next week",
                   "status": "not started"})
        store.put(todo_namespace,
                  "parents_todo",
                  {"task": "Call parents back about Thanksgiving plans",
                   "deadline": None,
                   "status": "not started"})
        todo_memories = list(store.search(todo_namespace))
    current_todos = format_todos(todo_memories)
    operation_message = SystemMessage(content=(f"{TODO_SYSTEM_MESSAGE}\n\n"
                                               f"Current ToDos:\n{current_todos}"))

    structured_model = model.with_structured_output(TodoRequest)
    request = structured_model.invoke([operation_message] + state["messages"])

    if request.action == "list":
        answer_message = SystemMessage(content=(
                "Answer the user's question using only the following "
                "ToDo records. Do not claim that a ToDo was changed.\n\n"
                f"{current_todos}"))

        response = model.invoke([answer_message] + state["messages"])
        return {"messages": [response]}

    if request.action == "add":
        if not request.task or not request.task.strip():
            return {"messages": [AIMessage(content=(
                            "What task would you like me to add?"))]}

        todo_id = f"todo_{uuid4().hex[:8]}"

        new_todo = {"task": request.task.strip(),
                    "deadline": (request.deadline.strip()
                                 if request.deadline
                                 else None),
                    "status": (request.status.strip()
                               if request.status
                               else "not started")}

        store.put(todo_namespace, todo_id, new_todo)

        return {"messages": [AIMessage(content=("ToDo added successfully.\n\n"
                                                f"- ID: {todo_id}\n"
                                                f"- Task: {new_todo['task']}\n"
                                                f"- Deadline: "
                                                f"{new_todo['deadline'] or 'No deadline'}\n"
                                                f"- Status: {new_todo['status']}"))]}

    selected_todo, lookup_error = find_todo(todo_memories, request.target, request.target_position)

    if lookup_error:
        return {"messages": [AIMessage(content=lookup_error)]}

    if request.action == "update":
        updated_todo = dict(selected_todo.value)
        changed_fields = []

        if request.task is not None:
            updated_todo["task"] = request.task.strip()
            changed_fields.append("task")

        if request.deadline is not None:
            updated_todo["deadline"] = request.deadline.strip()
            changed_fields.append("deadline")

        if request.status is not None:
            updated_todo["status"] = request.status.strip()
            changed_fields.append("status")

        if not changed_fields:
            return {"messages": [AIMessage(content=(
                            "What would you like to change about "
                            f'"{updated_todo.get("task", "this ToDo")}"?'))]}

        store.put(todo_namespace, selected_todo.key, updated_todo)

        return {"messages": [AIMessage(content=("ToDo updated successfully.\n\n"
                                                f"- ID: {selected_todo.key}\n"
                                                f"- Task: {updated_todo.get('task')}\n"
                                                f"- Deadline: "
                                                f"{updated_todo.get('deadline') or 'No deadline'}\n"
                                                f"- Status: "
                                                f"{updated_todo.get('status', 'not started')}"))]}

    store.delete(todo_namespace, selected_todo.key)

    return {"messages": [AIMessage(content=("ToDo deleted successfully:\n\n"
                                            f"- {selected_todo.value.get('task', 'Untitled')}"))]}
    

builder = StateGraph(State)

builder.add_node("router", router)
builder.add_node("calculator", calculator)
builder.add_node("tools", ToolNode(tools))
builder.add_node("search_web", search_web)
builder.add_node("answer_search", answer_search)
builder.add_node("database_lookup", database_lookup)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_request,)

builder.add_conditional_edges("calculator",tools_condition,)
builder.add_edge("tools", "calculator")

builder.add_edge("search_web", "answer_search")
builder.add_edge("answer_search", END)

builder.add_edge("database_lookup", END)


across_thread_memory = InMemoryStore()
within_thread_memory = MemorySaver()

local_graph = builder.compile().with_config(
    run_name="Multi-Node LangGraph Exercise 3 - Local")


user_id = "Test"

across_thread_memory.put(("todo", user_id),
                          "travel_todo",
                         {"task": "Finish booking travel to Hong Kong",
                          "deadline": "End of next week",
                          "status": "not started"})

across_thread_memory.put(("todo", user_id),
                          "parents_todo",
                         {"task": "Call parents back about Thanksgiving plans",
                          "deadline": None,
                          "status": "not started"})
