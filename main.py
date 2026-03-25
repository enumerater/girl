from langchain_core.messages import HumanMessage
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

# 主程序循环
while True:
    user_input = input("User: ")
    if user_input in ("exit", "quit", "bye"):
        break

    # 流式调用智能体
    for chunk in agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode=["updates", "custom"],
            version="v2",
    ):
        mode, data = chunk

        if mode == "updates":
            for node_name, state in data.items():
                # ====================== 核心提取逻辑 ======================
                if node_name == "recall":
                    # 提取回忆上下文
                    real_context = state.get("real_context", "无上下文")
                    print(f"\n📌 Node recall 核心信息：")
                    print(f"  回忆上下文：{real_context}")

                elif node_name == "llm_call":
                    msg = state.get("messages", [None])[0]
                    if msg:
                        ai_content = msg.content
                        meta = msg.response_metadata  # 元数据在消息对象内！
                        model = meta.get("model_name", "未知模型")
                        token = meta.get("token_usage", {})

                        print(f"\n📌 Node llm_call 核心信息：")
                        print(f"  AI回复：{ai_content}")
                        print(f"  使用模型：{model}")
                        print(
                            f"  Token消耗 → 输入：{token.get('input_tokens', 0)}  输出：{token.get('output_tokens', 0)}  总计：{token.get('total_tokens', 0)}")

                elif node_name == "memorize":
                    # 提取记忆内容
                    save_memory = state.get("save_memory", "无记忆")
                    print(f"\n📌 Node memorize 核心信息：")
                    print(f"  存储记忆：{save_memory}")

                # 其他节点保持默认输出
                else:
                    print(f"\nNode {node_name} updated: {state}")

        elif mode == "custom":
            print(f"Status: {data['status']}")
