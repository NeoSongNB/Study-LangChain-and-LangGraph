# 17｜使用Callbacks观察执行事件
# 
# 【学习目标】通过 BaseCallbackHandler 订阅运行事件，并对照下一份中间件示例。
# 【阅读顺序】继承回调类 → 实现事件方法 → 在 config.callbacks 中注册 → 消费 stream。
# 【易错提醒】原回调实现有输入类型假设；嵌套运行可能传入字典、None 或非 messages 输出，详见方法旁注释。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv('ARK_API_KEY')

# --- 1. 定义生命周期钩子处理器 ---
# Callbacks 用于订阅链、模型等运行事件；不是 18 中 middleware 钩子的别名。
# 原代码日志里的 before_agent 等文字只是类比标签，不代表真实中间件被调用。
class LifecycleHandler(BaseCallbackHandler):
    # 原实现假设 inputs 是字典，messages 非空且元素有 .content，实际嵌套事件不一定满足。
    # 建议修复时先检查输入类型和消息类型，并按 run_id/parent_run_id 区分顶层与子节点。
    def on_chain_start(self, serialized, inputs, **kwargs):
        """对应 before_agent: 代理整体流程开始"""
        print(f"\n🚀 [before_agent] 代理准备出发！输入内容: {inputs.get('messages', [])[-1].content}")

    # 聊天模型还有专门的 on_chat_model_start 回调；不能假设本方法覆盖所有聊天模型开始事件。
    def on_llm_start(self, serialized, prompts, **kwargs):
        """对应 before_model: 模型调用前"""
        print(f"🧠 [before_model] 正在思考中... (准备调用模型)")

    def on_llm_end(self, response, **kwargs):
        """对应 after_model: 模型响应后"""
        # response 是 LLMResult 对象，包含生成的文本或工具调用
        print(f"✨ [after_model] 模型给出了回应。")

    # 链结束事件也可能来自嵌套运行；outputs 未必是包含 messages 的字典。
    # 此处保留原代码供对照；希望按 Agent 边界稳定打印时，优先参考 18 的中间件。
    def on_chain_end(self, outputs, **kwargs):
        """对应 after_agent: 代理整体流程结束"""
        print(f"🏁 [after_agent] 任务完成！最终输出: {outputs.get('messages', [])[-1].content}")

# --- 2. 基础配置与工具定义 ---
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
    return f"{city} 天气晴朗，阳光明媚，非常适合出门玩耍！"

# ToolRuntime[Context] 由框架注入，不需要模型猜测或传入 runtime 参数。
# 函数通过 runtime.context 读取应用提供的用户 ID；此例只用条件表达式模拟定位。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 检索用户信息。"""
    user_id = runtime.context.user_id
    return "佛罗里达" if user_id == "1" else "旧金山"

SYSTEM_PROMPT = "你是一位精通天气的专家。用户问天气时，先用 get_user_location 获取位置，再查天气。"

# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 创建代理
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

# --- 3. 运行并注入回调 ---
config = {
    "configurable": {"thread_id": "1"},
    "callbacks": [LifecycleHandler()]  # 在这里注入生命周期钩子
}
ctx = Context(user_id="1")

print("--- 开始执行代理 ---")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "今天什么天气？"}]},
    config=config,
    context=ctx,
    # values 每次给出当前完整状态；外层是状态字段，例如 messages，而不是节点名称。
    stream_mode="values"
):
    # 必须消费 stream 迭代器才会推进执行；pass 表示不额外处理数据，日志交给回调。
    pass # 具体的打印逻辑已由 LifecycleHandler 处理