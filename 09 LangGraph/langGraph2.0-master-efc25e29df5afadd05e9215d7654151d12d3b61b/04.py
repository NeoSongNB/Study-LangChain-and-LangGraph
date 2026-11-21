"""
LangGraph 1.0+ 教程 - 04.py
LLM + 天气查询工具 + Checkpointer 最简单示例
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
import json

# 加载 .env 文件
load_dotenv()

# ============================================================
# Step 1: LLM 配置
# ============================================================
api_key = os.environ.get("ARK_API_KEY")
llm = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


# ============================================================
# Step 2: 定义天气查询工具
# ============================================================
@tool
def get_weather(city: str) -> str:
    """查询城市天气

    Args:
        city: 城市名称
    """
    # 模拟天气数据
    weather_data = {
        "北京": "北京今天天气晴朗，温度 25°C",
        "上海": "上海今天多云，温度 22°C",
        "广州": "广州今天下雨，温度 28°C",
        "深圳": "深圳今天晴天，温度 29°C",
    }
    return weather_data.get(city, f"{city}的天气情况未知")


# 绑定工具到 LLM
tools = [get_weather]
tools_by_name = {t.name: t for t in tools}
model_with_tools = llm.bind_tools(tools)


# ============================================================
# Step 3: 定义状态
# ============================================================
class WeatherState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    city: str
    weather_result: str


# ============================================================
# Step 4: 定义节点
# ============================================================
def call_llm(state: WeatherState):
    """LLM 判断是否需要调用天气工具"""
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content="你是一个天气助手。如果用户询问天气，调用 get_weather 工具；"
                        "否则直接回复用户。"
            )
        ]
        + state["messages"]
    )
    return {"messages": [response]}


def get_weather_node(state: WeatherState):
    """执行天气查询工具"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_obj = tools_by_name[tool_call["name"]]
        observation = tool_obj.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        # 同时保存天气结果到 state
        if tool_call["name"] == "get_weather":
            return {"weather_result": observation, "messages": result}
    return {"messages": result}


def final_response(state: WeatherState):
    """最终回复用户"""
    return {"messages": [AIMessage(content="天气查询完成，请查看上方结果。")]}


# ============================================================
# Step 5: 定义条件判断
# ============================================================
def should_continue(state: WeatherState) -> Literal["get_weather", END]:
    """根据 LLM 是否调用工具决定下一步"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "get_weather"
    return END


# ============================================================
# Step 6: 构建图
# ============================================================
def main():
    # 构建工作流
    graph = StateGraph(WeatherState)

    # 添加节点
    graph.add_node("call_llm", call_llm)
    graph.add_node("get_weather", get_weather_node)
    graph.add_node("final_response", final_response)

    # 添加边
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        ["get_weather", END],
    )
    graph.add_edge("get_weather", "final_response")
    graph.add_edge("final_response", END)

    # 添加 checkpointer
    checkpointer = InMemorySaver()
    agent = graph.compile(checkpointer=checkpointer)

    # ============================================================
    # Step 7: 调用并查看 checkpointer
    # ============================================================
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    print("=== LangGraph 天气查询示例 ===\n")

    # 第一次调用
    print("--- 调用：查询北京天气 ---")
    messages = [HumanMessage(content="北京今天天气怎么样？")]
    result = agent.invoke({"messages": messages}, config)
    for m in result["messages"]:
        m.pretty_print()

    print("\n" + "=" * 60)
    print("=== Checkpointer 存储内容 ===")
    print("=" * 60)

    checkpoint_tuple = checkpointer.get_tuple(config)
    if checkpoint_tuple:
        print(f"Thread ID: {checkpoint_tuple.config['configurable']['thread_id']}")
        print(f"Checkpoint ID: {checkpoint_tuple.checkpoint['id']}")
        print(f"Timestamp: {checkpoint_tuple.checkpoint['ts']}")
        print()
        print("=== Channel Values (状态) ===")
        channel_values = checkpoint_tuple.checkpoint["channel_values"]
        for key, value in channel_values.items():
            if key == "messages":
                print(f"  {key}: [{len(value)} 条消息]")
                for i, msg in enumerate(value):
                    msg_type = getattr(msg, 'type', 'dict') if isinstance(msg, dict) else msg.type
                    msg_content = msg.get('content', '') if isinstance(msg, dict) else msg.content
                    print(f"    [{i}] {msg_type}: {msg_content[:50]}...")
            else:
                print(f"  {key}: {value}")
        print()
        print("=== 完整 Checkpoint JSON ===")
        # 转换 messages 为可 JSON 序列化的格式
        checkpoint_json = checkpoint_tuple.checkpoint.copy()
        channel_values = checkpoint_json["channel_values"]
        if "messages" in channel_values:
            channel_values["messages"] = [
                {"type": msg.type, "content": msg.content} if hasattr(msg, 'type') else msg
                for msg in channel_values["messages"]
            ]
        print(json.dumps(checkpoint_json, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("=== 流程执行顺序 ===")
    print("  START -> call_llm -> (判断) -> get_weather -> final_response -> END")
    print("=" * 60)


if __name__ == "__main__":
    main()

