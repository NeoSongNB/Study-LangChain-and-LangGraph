# 19｜六类中间件钩子与调用耗时监控
# 
# 【学习目标】组合节点钩子和环绕钩子，记录模型耗时并观察工具执行。
# 【阅读顺序】Agent 开始 → 模型前置钩子 → 环绕模型调用 → 模型后置钩子 → 按需环绕工具 → 结束。
# 【易错提醒】原有工具名读取与异常返回存在问题，已在对应位置标出建议修复；本次保留原演示逻辑。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
import time
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent, 
    after_agent, 
    before_model, 
    after_model, 
    wrap_model_call,
    wrap_tool_call, 
    AgentState
)
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

# 加载环境变量
# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()

# --- 1. 定义中间件钩子 ---

@before_agent
def on_agent_start(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    print("\n🚀 [1. before_agent] 代理引擎启动...")
    return None

# 环绕钩子可以在 next_call 前后增加逻辑；必须返回模型响应以继续 Agent 流程。
@wrap_model_call
def monitor_model(request: Any, next_call: Any) -> Any:
    """环绕模型调用：计算模型响应耗时"""
    # 这里的 request.state 包含了当前所有的消息记录
    print(f"⏳ [2. wrap_model_call] (进入) 正在请求 LLM...")
    # 本例 time.time() 测量墙上时钟差；严谨的耗时测量更适合 time.perf_counter()。
    start_time = time.time()
    
    # 执行真正的模型请求
    response = next_call(request)
    
    duration = time.time() - start_time
    print(f"⌛ [2. wrap_model_call] (退出) LLM 响应完成，耗时: {duration:.2f}s")
    return response

@before_model(can_jump_to=["end"])
def on_model_start(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    print("🧠 [3. before_model] 准备进入模型节点...")
    return None

@after_model
def on_model_complete(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    print(f"✨ [4. after_model] 模型输出已存入状态。")
    return None

@wrap_tool_call
def secure_tool_executor(request: Any, next_call: Any) -> Any:
    """环绕工具调用：拦截或审计工具执行"""
    # 原代码问题：工具名应从 request.tool_call["name"] 读取，state 不是当前工具调用描述。
    # 这行保留原样；若未额外向 state 写入 tool_name，日志通常只会显示 unknown。
    tool_name = request.state.get("tool_name", "unknown")
    print(f"🛡️ [5. wrap_tool_call] (进入) 安全审计工具: {tool_name}")
    
    try:
        # 执行真正的工具调用
        result = next_call(request)
        print(f"✅ [5. wrap_tool_call] (退出) 工具执行成功")
        return result
    except Exception as e:
        print(f"❌ [5. wrap_tool_call] (退出) 工具执行异常: {e}")
        # 原代码问题：wrap_tool_call 的异常分支应返回 ToolMessage 或 Command，而不是裸字符串。
        # 建议返回 ToolMessage(content=str(e), tool_call_id=request.tool_call["id"])，并导入 ToolMessage。
        # 工具函数内部返回 str 是正常的；此处讨论的是环绕中间件本身的返回类型。
        return f"工具调用发生错误: {e}"

@after_agent
def on_agent_end(state: AgentState, runtime: ToolRuntime) -> dict[str, Any] | None:
    print("\n🏁 [6. after_agent] 代理所有任务已结束。")
    return None

# --- 2. 工具与环境配置 ---

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
    print(f"🏃 [Tool Execution] 正在查询 {city} 的实时天气...")
    return f"{city} 天气晴朗，非常适合出门！"

# ToolRuntime[Context] 由框架注入，不需要模型猜测或传入 runtime 参数。
# 函数通过 runtime.context 读取应用提供的用户 ID；此例只用条件表达式模拟定位。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取地理位置。"""
    user_id = runtime.context.user_id
    location = "佛罗里达" if user_id == "1" else "旧金山"
    print(f"🏃 [Tool Execution] 定位用户 {user_id} 为: {location}")
    return location

# --- 3. 初始化代理并注册中间件 ---

# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=os.getenv('ARK_API_KEY'),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 注册 6 个钩子
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    system_prompt="你是一位幽默的天气专家。请先获取用户位置，再查天气。",
    # 注册可供模型选择的工具；列表顺序不保证执行顺序，具体调用由模型和业务逻辑决定。
    tools=[get_user_location, get_weather_for_location],
    # context_schema 声明运行时上下文结构；实际数据仍需在 invoke/stream 的 context 参数中传入。
    context_schema=Context,
    # 注册列表中的数字是原演示标签，不是严格的时序编号。
    # before_model 先于真正的模型请求，wrap_model_call 包裹该请求；工具循环可能重复多轮。
    middleware=[
        on_agent_start,
        monitor_model,      # wrap_model_call
        on_model_start,
        on_model_complete,
        secure_tool_executor, # wrap_tool_call
        on_agent_end
    ]
)

# --- 4. 执行流程 ---

# 只在直接运行当前脚本时执行下面的入口代码；被其他文件 import 时跳过此代码块。
if __name__ == "__main__":
    # thread_id 用于标识会话；只有配置了 checkpointer 才会据此保存和恢复线程状态。
    # 仅传入这个配置不会自动启用跨 invoke 的记忆。
    config = {"configurable": {"thread_id": "weather_test_001"}}
    ctx = Context(user_id="1")

    print("--- 启动对话流 ---")
    inputs = {"messages": [{"role": "user", "content": "帮我看看我现在这儿天气咋样？"}]}
    
    for chunk in agent.stream(
        inputs,
        config=config,
        context=ctx,
        # values 每次给出当前完整状态；外层是状态字段，例如 messages，而不是节点名称。
        stream_mode="values"
    ):
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]
            # 在 LangChain 中，AIMessage 的 type 为 'ai'
            # 这里仅筛选有正文的 AI 消息，不保证它就是最后一次回答；模型也可能边输出文字边请求工具。
            if last_msg.content and last_msg.type == "ai":
                print(f"\n📢 [最终回答]: {last_msg.content}")

    print("\n--- 演示结束 ---")