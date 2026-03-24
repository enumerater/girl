from langchain.messages import SystemMessage, trim_messages
from model import model, vector_store
from tool_model import model_with_tools, tools_by_name

# L1 memory: Trim messages to keep only the last 10
# This is a basic sliding window
trimmer = trim_messages(
    max_tokens=20, # 这里的 10 只是为了演示，实际可根据 token 数调整
    strategy="last",
    token_counter=len,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def recall(state: dict):
    """Retrieve L2/L3 memories based on the latest user message"""
    last_user_message = state["messages"][-1].content

    # 检索最相关的 2 条历史记录
    docs = vector_store.similarity_search(last_user_message, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    return {"context": context}

def llm_call(state: dict):
    """LLM decides whether to call a tool or not, using L1/L2/L3 context"""

    # L1: 使用裁剪后的消息
    trimmed_messages = trimmer.invoke(state["messages"])

    # L2/L3: 构造增强的系统提示词
    memory_context = f"\n你回想起的相关记忆：\n{state.get('context', '无')}"
    system_msg = SystemMessage(
        content=f"你现在扮演用户的女朋友，负责提供情绪价值。{memory_context}\n如果用户问到你不知道的信息，请务必先查记忆或调用工具。"
    )

    return {
        "messages": [
            model_with_tools.invoke([system_msg] + trimmed_messages)
        ]
    }

from langchain.messages import ToolMessage

def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def memorize(state: dict):
    """Save the latest exchange as L2/L3 memory for the future"""
    if len(state["messages"]) >= 2:
        last_human = state["messages"][-2].content
        last_ai = state["messages"][-1].content
        memory_str = f"用户问：{last_human}\n你回答：{last_ai}"

        # 将这次对话存入向量库
        vector_store.add_texts([memory_str])
    return {}

from typing import Literal
from langgraph.graph import END, MessagesState

def should_continue(state: MessagesState) -> Literal["tool_node", "memorize"]:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    # 回答完毕后，去进行记忆存储
    return "memorize"