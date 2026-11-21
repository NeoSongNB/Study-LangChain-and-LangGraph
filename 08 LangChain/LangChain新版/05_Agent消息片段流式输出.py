# 05｜Agent消息片段流式输出
# 
# 【学习目标】观察 Agent 的 messages 流中每个消息片段及其元数据。
# 【阅读顺序】创建无工具 Agent → stream_mode="messages" → 逐项打印返回的二元组。
# 【易错提醒】每项为 (消息片段, 元数据)，不是完整状态字典；这与 08 的 values 模式不同。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import json
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pprint import pprint
from dotenv import load_dotenv

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv("ARK_API_KEY")

# Define system prompt
SYSTEM_PROMPT = "你是智能助手小宁"

# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# Create agent (no tools = direct LLM responses, but streaming works)
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
)

print("Streaming response:")
# Stream agent execution
# messages 模式每次产出 (AIMessageChunk, metadata) 二元组。
# 可在循环中用 message_chunk, metadata = chunk 解包，并查看 message_chunk.content。
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "罗永浩帅吗？"}]}, stream_mode="messages"
):
    # print('==============')
    # pprint(chunk)
    print("----------------------------------\n")
    print(json.dumps(chunk, indent=2, ensure_ascii=False, default=lambda x: str(x)))

    # 下面保留的注释代码属于 values 模式的访问方式，不能直接用于本例 messages 模式。
    # 以下保留为 values 模式对照；本例 messages 模式不能用这种字典访问方式。
    # latest_message = chunk["messages"][-1]
    # if latest_message.content:
    #     print(f"Agent: {latest_message.content}")
    # elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
    #     print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

