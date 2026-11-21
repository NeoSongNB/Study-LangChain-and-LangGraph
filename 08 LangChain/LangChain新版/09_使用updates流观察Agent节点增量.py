# 09｜使用updates流观察Agent节点增量
# 
# 【学习目标】观察模型节点与工具节点各自写入状态的消息增量。
# 【阅读顺序】stream_mode="updates" → 遍历节点名 → 读取节点更新 → 打印其中的消息。
# 【易错提醒】updates 的外层是节点名，内层是该节点本次更新；不要按 values 的结构访问。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv('ARK_API_KEY')

SYSTEM_PROMPT = """你是一位精通天气预报的专家，说话喜欢用谐音梗或幽默的口吻。

你可以使用以下两个工具：
- get_weather_for_location: 用于查询特定城市的天气。
- get_user_location: 用于获取当前用户所在的地理位置。

如果用户询问天气，请确保你知道具体地点。如果从问题中可以看出用户指的是他们所在地，请先使用 get_user_location 工具获取位置。"""

# dataclass 自动生成初始化方法，因此可以写 Context(user_id="1")。
# 这个上下文由应用提供，适合放用户身份等依赖；它不等同于 messages 对话历史。
@dataclass
class Context:
    user_id: str

# @tool 根据函数名、类型注解和文档字符串生成模型可见的工具说明。
# city 是模型填写的业务参数；返回的天气字符串会被框架包装成工具结果消息。
# 注意：本函数没有请求气象 API，返回的是用于学习的模拟数据。
@tool
def get_weather_for_location(city: str) -> str:
    """获取给定城市的天气情况。
    
    Args:
        city: 城市名称，例如 "北京" 或 "上海"
    """
    return f"{city} 天气晴朗，阳光明媚，非常适合出门玩耍！"

# ToolRuntime[Context] 由框架注入，不需要模型猜测或传入 runtime 参数。
# 函数通过 runtime.context 读取应用提供的用户 ID；此例只用条件表达式模拟定位。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 检索用户当前位置。
    
    Args:
        runtime: 运行时上下文，包含用户ID
    """
    user_id = runtime.context.user_id
    return "佛罗里达" if user_id == "1" else "旧金山"

# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

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
ctx = Context(user_id="1")

print("Streaming agent UPDATES (deltas only):\n")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "今天什么天气？"}]},
    config=config,
    context=ctx,
    # updates 每次给出节点产生的状态增量，形如 {节点名: {"messages": [...]}}。
    stream_mode="updates"
):
    # node_name 常见为 model 或 tools；node_update 只包含该节点本次写入的字段。
    for node_name, node_update in chunk.items():
        print(f"Node: {node_name}")
        print(json.dumps(node_update, indent=2, ensure_ascii=False, default=lambda x: str(x)))
        print("\n")

        if "messages" in node_update:
            # 一个节点更新里可能有多条消息，因此逐条展示，不要总是假定只有一条。
            for msg in node_update["messages"]:
                msg_type = msg.__class__.__name__
                if msg.content:
                    print(f"[{node_name} | {msg_type}] {msg.content}")
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tools = [tc["name"] for tc in msg.tool_calls]
                    print(f"[{node_name} | {msg_type}] Calling tools: {tools}")
                else:
                    print(f"[{node_name} | {msg_type}]")

