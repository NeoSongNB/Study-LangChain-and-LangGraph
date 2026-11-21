# 18｜使用中间件观察Agent生命周期
# 
# 【学习目标】用 before/after 钩子观察 Agent 开始结束与每次模型调用。
# 【阅读顺序】定义四个中间件 → 注册 middleware → 执行工具循环 → 观察日志。
# 【易错提醒】一次 Agent 执行可能调用模型多次，所以 before_model/after_model 可能反复触发。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
from dataclasses import dataclass
from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()

# --- 1. 定义中间件钩子 ---

# before_agent 在本次 Agent 执行开始时运行，适合初始化与入口日志。
@before_agent
def on_agent_start(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    print("\n🚀 [before_agent] 代理引擎启动...")
    return None

# before_model 在每次模型调用前执行；can_jump_to 只声明允许的跳转目标。
# 本函数返回 None，没有返回 jump_to 指令，因此不会提前终止 Agent。
@before_model(can_jump_to=["end"])
def on_model_start(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    print("\n🧠 [before_model] 准备询问模型下一步该做什么...")
    return None

# after_model 在模型输出写入状态后执行，可读取 state["messages"][-1]。
# 此刻的 tool_calls 是工具请求；真正的工具函数可能还没有执行。
@after_model
def on_model_complete(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    last_msg = state["messages"][-1]
    print(f"✨ [after_model] 模型决策完成。")
    
    # 打印工具调用详情
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            print(f"🛠️  [Tool Call] 正在调用工具: {tc['name']} | 参数: {tc['args']}")
    elif last_msg.content:
        print(f"💬 [Final Response] 模型直接回复了文字。")
    return None

# after_agent 在本次 Agent 正常完成时运行，不等同于捕获所有异常的 finally。
@after_agent
def on_agent_end(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    print("\n🏁 [after_agent] 代理执行完毕，清理现场。")
    return None

# --- 2. 工具定义 (带内部打印) ---

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
    """获取给定城市的天气情况。"""
    print(f"🏃 [Executing] 正在查询 {city} 的天气数据库...")
    return f"{city} 天气晴朗，心情指数 100 分！"

# ToolRuntime[Context] 由框架注入，不需要模型猜测或传入 runtime 参数。
# 函数通过 runtime.context 读取应用提供的用户 ID；此例只用条件表达式模拟定位。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 检索用户信息。"""
    user_id = runtime.context.user_id
    loc = "佛罗里达" if user_id == "1" else "旧金山"
    print(f"🏃 [Executing] 正在根据 UserID:{user_id} 定位到: {loc}")
    return loc

# --- 3. 初始化代理 ---

# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=os.getenv('ARK_API_KEY'),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    system_prompt="你是一位幽默的天气专家。用户问天气时，请务必先用 get_user_location 获取位置，再用 get_weather_for_location 查天气。",
    # 注册可供模型选择的工具；列表顺序不保证执行顺序，具体调用由模型和业务逻辑决定。
    tools=[get_user_location, get_weather_for_location],
    # context_schema 声明运行时上下文结构；实际数据仍需在 invoke/stream 的 context 参数中传入。
    context_schema=Context,
    middleware=[on_agent_start, on_model_start, on_model_complete, on_agent_end]
)

# --- 4. 执行测试 ---

# thread_id 用于标识会话；只有配置了 checkpointer 才会据此保存和恢复线程状态。
# 仅传入这个配置不会自动启用跨 invoke 的记忆。
config = {"configurable": {"thread_id": "weather_chat_1"}}
ctx = Context(user_id="1")

print("--- 开始运行 ---")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "帮我看看我这儿天气咋样？"}]},
    config=config,
    context=ctx,
    # values 每次给出当前完整状态；外层是状态字段，例如 messages，而不是节点名称。
    stream_mode="values"
):
    # 这里的 pass 不打印状态；终端输出来自中间件和工具函数内部。
    # stream_mode="values" 产出完整状态，但本循环未直接打印状态。
    pass