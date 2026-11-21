"""
LangGraph 1.0+ 基础教程 - 02.py
演示如何使用真实的 LLM 构建图
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

# 加载 .env 文件
load_dotenv()

# 从 main.py 移植过来的 LLM 配置
api_key = os.environ.get("ARK_API_KEY")
llm = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


def call_llm(state: MessagesState):
    """调用 LLM 的节点"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def main():
    # 1. 创建 StateGraph，使用 MessagesState 作为状态类型
    graph = StateGraph(MessagesState)

    # 2. 添加节点（使用真实 LLM）
    graph.add_node("call_llm", call_llm)

    # 3. 添加边：START -> call_llm -> END
    graph.add_edge(START, "call_llm")
    graph.add_edge("call_llm", END)

    # 4. 编译图
    compiled_graph = graph.compile()

    # 5. 调用图
    result = compiled_graph.invoke({"messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]})

    print("=== LangGraph 调用结果 ===")
    print(result)
    print("\n=== 消息内容 ===")
    for msg in result["messages"]:
        # langGraph 1.0+ 返回的是 Message 对象，有 type 和 content 属性
        print(f"{msg.type}: {msg.content}")


if __name__ == "__main__":
    main()

