from langchain.tools import tool

from model import model


@tool
def send_pic() -> str:
    """
    当需要发送表情包时，调这个工具。
    """
    return "用户叫阿伟"


tools = [send_pic]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)