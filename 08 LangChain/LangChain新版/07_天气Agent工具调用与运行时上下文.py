# 07｜天气Agent工具调用与运行时上下文
# 
# 【学习目标】让模型选择定位、天气工具，并由程序注入用户 ID。
# 【阅读顺序】定义 Context → 注册两个工具 → create_agent → invoke 时传入用户上下文。
# 【易错提醒】天气和位置都是固定模拟数据；虽然导入了 InMemorySaver，本例并未启用记忆。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json


# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv('ARK_API_KEY')

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# Define context schema
# dataclass 自动生成初始化方法，因此可以写 Context(user_id="1")。
# 这个上下文由应用提供，适合放用户身份等依赖；它不等同于 messages 对话历史。
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
# @tool 根据函数名、类型注解和文档字符串生成模型可见的工具说明。
# city 是模型填写的业务参数；返回的天气字符串会被框架包装成工具结果消息。
# 注意：本函数没有请求气象 API，返回的是用于学习的模拟数据。
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# ToolRuntime[Context] 由框架注入，不需要模型猜测或传入 runtime 参数。
# 函数通过 runtime.context 读取应用提供的用户 ID；此例只用条件表达式模拟定位。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


api_key = os.environ["ARK_API_KEY"]
# 将已读取的密钥传给客户端，无需设置 OPENAI_API_KEY。
# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=api_key,                    # 👈 直接传入API Key
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 👈 直接传入Base URL
)

# Create agent
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    # 注册可供模型选择的工具；列表顺序不保证执行顺序，具体调用由模型和业务逻辑决定。
    tools=[get_user_location, get_weather_for_location],
    # context_schema 声明运行时上下文结构；实际数据仍需在 invoke/stream 的 context 参数中传入。
    context_schema=Context,
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
# thread_id 用于标识会话；只有配置了 checkpointer 才会据此保存和恢复线程状态。
# 仅传入这个配置不会自动启用跨 invoke 的记忆。
config = {"configurable": {"thread_id": "1"}}

# 预期流程：询问当地天气 → 定位用户 → 查该地点天气 → 模型整合结果。
# Context(user_id="1") 由程序构造，不需要出现在用户问题文本里。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "今天什么天气？"}]},
    config=config,
    context=Context(user_id="1")
)

print("----------------------------------\n")
print(json.dumps(response, indent=2, ensure_ascii=False, default=lambda x: str(x)))
# pprint(response)