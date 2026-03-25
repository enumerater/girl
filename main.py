from langgraph.constants import START, END
from langgraph.graph import StateGraph
from node import llm_call, tool_node, should_continue, recall, memorize
from state import MessagesState

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("recall", recall)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("memorize", memorize)

# Connect nodes
agent_builder.add_edge(START, "recall")
agent_builder.add_edge("recall", "llm_call")

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "memorize": "memorize"
    }
)

agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("memorize", END)

# Compile the agent
agent = agent_builder.compile()

# Test the multi-level memory architecture
from langchain.messages import HumanMessage

def chat(text):
    print(f"\n--- User: {text} ---")
    global messages
    messages.append(HumanMessage(content=text))
    result = agent.invoke({"messages": messages})
    # Update messages in the outer scope
    messages = result["messages"]
    # Print the last AI message
    print("--- AI Message ---")
    messages[-1].pretty_print()

messages = []

chat("用户叫什么")
