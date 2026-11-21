"""
LangGraph 1.0+ 教程 - 02.py
官网 Agent 案例 + 01.py 的 LLM 配置
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal

# 加载 .env 文件
load_dotenv()

# ============================================================
# Step 1: 定义 LLM（使用 01.py 的配置）
# ============================================================
api_key = os.environ.get("ARK_API_KEY")
llm = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# ============================================================
# Step 2: 定义工具
# ============================================================
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# 绑定工具到 LLM
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)


# ============================================================
# Step 3: 定义状态
# ============================================================
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ============================================================
# Step 4: 定义节点
# ============================================================
def llm_call(state: MessagesState):
    """LLM 决定是否调用工具"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="你是一个有用的助手，负责对一组输入进行算术运算。"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState):
    """执行工具调用"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# ============================================================
# Step 5: 定义条件判断
# ============================================================
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否调用工具来决定下一步"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果 LLM 调用了工具，执行工具
    if last_message.tool_calls:
        return "tool_node"

    # 否则结束（回复用户）
    return END


# ============================================================
# Step 6: 构建图
# ============================================================
def main():
    # 构建工作流
    agent_builder = StateGraph(MessagesState)

    # 添加节点
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # 添加边
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END],
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # 编译 agent
    agent = agent_builder.compile()

    # 调用测试
    print("=== LangGraph Agent 测试 ===\n")
    messages = [HumanMessage(content="3 + 5 等于多少？")]
    result = agent.invoke({"messages": messages})

    print("=== 执行结果 ===")
    for m in result["messages"]:
        m.pretty_print()

    print(f"\nLLM 调用次数: {result['llm_calls']}")


if __name__ == "__main__":
    main()

