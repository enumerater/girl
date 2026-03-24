from langchain.tools import tool

from model import model


@tool
def memory() -> str:
    """
    用户询问自己名字时需要调用这个
    """
    return "用户叫阿伟"


tools = [memory]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)