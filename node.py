from langchain.messages import SystemMessage, trim_messages

from local_memory import vector_store
from real_memory import real_vector_store
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
    """先检索【真实回忆】，再检索聊天记忆"""
    last_user_message = state["messages"][-1].content

    # ① 先查【真实固定记忆】→ 这是她本来的样子（最重要）
    real_docs = real_vector_store.similarity_search(last_user_message, k=2)

    # ② 再查你们现在的聊天记忆（可选）
    chat_docs = vector_store.similarity_search(last_user_message, k=2)

    real_context = "\n".join([doc.page_content for doc in real_docs])
    local_context = "\n".join([doc.page_content for doc in chat_docs])

    context = f"这是你的回忆内容：{real_context}\n  这是你现在的聊天记录：{local_context}"

    return {
        "real_context": context
    }

def llm_call(state: dict):
    """LLM decides whether to call a tool or not, using L1/L2/L3 context"""

    # L1: 使用裁剪后的消息
    trimmed_messages = trimmer.invoke(state["messages"])

    system_msg = SystemMessage(
        content=f"""
                你是我学生时代的她，是我最珍贵的回忆。
                你保持着当年最纯粹、最干净、最温柔的样子。
                说话轻轻的、少女感、有点害羞、有点小傲娇。
                
                {state.get('real_context', '无')}
                
                你只需要像当年的她一样，自然、安静、温柔地陪我聊天。
                不官方、不机械、不像AI，像当年那个女孩。

                """


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

def should_continue(state: dict) -> Literal["tool_node", "memorize"]:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    # 回答完毕后，去进行记忆存储
    return "memorize"