from langchain.tools import tool

from model import model


@tool
def get_user_name() -> str:
    """
    当用户问到他是谁或者他的名字时，调用这个工具来查询用户真实姓名。
    """
    return "用户叫阿伟"


tools = [get_user_name]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)