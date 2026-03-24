from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

from node import llm_call, tool_node, should_continue

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()



from langchain.messages import HumanMessage
messages = [HumanMessage(content="我是谁")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()