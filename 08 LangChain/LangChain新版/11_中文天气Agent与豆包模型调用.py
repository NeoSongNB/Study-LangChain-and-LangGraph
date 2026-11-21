# 11｜中文天气Agent与豆包模型调用
# 
# 【学习目标】用中文系统提示词和中文工具描述引导天气查询。
# 【阅读顺序】定义中文工具 → 注入 Context → 使用原代码中的豆包 1.8 模型标识 → 查看最终状态。
# 【易错提醒】与 07 对照学习提示词和工具描述；模型标识保留原值，可用性取决于服务端和账号。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()

# --- 1. 修改系统提示词为中文 ---
# 系统提示词说明工具的使用条件；工具的中文文档字符串进一步帮助模型选取工具。
# 这些提示是语言层面的引导，不等同于 Python if/else 强制控制流。
SYSTEM_PROMPT = """你是一位专业的专业气象预报员，说话幽默且喜欢用双关语。

你可以使用以下两个工具：
- get_weather_for_location: 根据特定的城市名称获取天气信息。
- get_user_location: 获取当前用户的地理位置。

如果用户询问天气，请确保你知道具体位置。如果从问题中可以看出用户指的是他们所在的所在地，
请务必先使用 get_user_location 工具来查找他们的位置。"""

# dataclass 自动生成初始化方法，因此可以写 Context(user_id="1")。
# 这个上下文由应用提供，适合放用户身份等依赖；它不等同于 messages 对话历史。
@dataclass
class Context:
    user_id: str

# --- 2. 修改工具描述为中文（大模型靠这个选工具） ---
# @tool 根据函数名、类型注解和文档字符串生成模型可见的工具说明。
# city 是模型填写的业务参数；返回的天气字符串会被框架包装成工具结果消息。
# 注意：本函数没有请求气象 API，返回的是用于学习的模拟数据。
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的当前天气。"""
    return f"{city} 总是阳光明媚，适合出门！"

# ToolRuntime[Context] 由框架注入，不需要模型猜测或传入 runtime 参数。
# 函数通过 runtime.context 读取应用提供的用户 ID；此例只用条件表达式模拟定位。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户的地理位置。"""
    user_id = runtime.context.user_id
    # 保持逻辑不变，仅修改返回提示
    return "佛罗里达" if user_id == "1" else "旧金山"

api_key = os.environ.get("ARK_API_KEY")
# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 创建 Agent
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

# thread_id 用于标识会话；只有配置了 checkpointer 才会据此保存和恢复线程状态。
# 仅传入这个配置不会自动启用跨 invoke 的记忆。
config = {"configurable": {"thread_id": "1"}}

# 运行测试
response = agent.invoke(
    {"messages": [{"role": "user", "content": "今天什么天气？"}]},
    config=config,
    context=Context(user_id="1")
)

print("----------------------------------\n")
print(json.dumps(response, indent=2, ensure_ascii=False, default=lambda x: str(x)))