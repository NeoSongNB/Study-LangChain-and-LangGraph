# 14｜流式观察工具决策与节点输出
# 
# 【学习目标】从节点更新中观察工具调用参数、消息内容和服务端可能返回的额外字段。
# 【阅读顺序】遍历节点更新 → 读取最后一条消息 → 查看 tool_calls、content 和 additional_kwargs。
# 【易错提醒】reasoning_content 仅在接口实际返回时存在；读取字段并不能取得模型未公开的内部推理。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()

# --- 时间追踪器 ---
# 这个辅助类只测量本地观察到的经过时间，不是模型服务内部的性能剖析器。
# 创建时记录基准点，mark 记录事件并更新上一次打点时刻。
class AgentTimer:
    def __init__(self):
        # perf_counter 适合测量时间间隔；其绝对数值不是日期时间。
        self.start_time = time.perf_counter()
        self.last_mark = self.start_time

    def mark(self, event_name):
        now = time.perf_counter()
        # elapsed_total 是距计时器创建的总时长；since_last 是两个打点之间的时长。
        elapsed_total = now - self.start_time
        since_last = now - self.last_mark
        print(f"\n⏱️ [{elapsed_total:.3f}s] {event_name}")
        self.last_mark = now

timer = AgentTimer()

# --- 1. 系统提示词 ---
SYSTEM_PROMPT = """你是一个天气查询助手。
请遵循以下确定性操作：
- 如果没有城市名，请立即执行 get_user_location。
- 如果已有城市名，请立即执行 get_weather_for_location。
在得到最终天气结果后，直接回复天气信息。"""

# dataclass 自动生成初始化方法，因此可以写 Context(user_id="1")。
# 这个上下文由应用提供，适合放用户身份等依赖；它不等同于 messages 对话历史。
@dataclass
class Context:
    user_id: str

# --- 2. 工具定义 ---
# ToolRuntime[Context] 由框架注入，不需要模型猜测或传入 runtime 参数。
# 函数通过 runtime.context 读取应用提供的用户 ID；此例只用条件表达式模拟定位。
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """获取用户当前城市(内部ID定位)。"""
    timer.mark("🛠️ 工具执行中: get_user_location")
    return "佛罗里达" if runtime.context.user_id == "1" else "旧金山"

# @tool 根据函数名、类型注解和文档字符串生成模型可见的工具说明。
# city 是模型填写的业务参数；返回的天气字符串会被框架包装成工具结果消息。
# 注意：本函数没有请求气象 API，返回的是用于学习的模拟数据。
@tool
def get_weather_for_location(city: str) -> str:
    """查询指定城市天气。"""
    timer.mark(f"🛠️ 工具执行中: get_weather_for_location (参数: {city})")
    return f"{city} 总是阳光明媚！"

# --- 3. 初始化 LLM ---
# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=0,
    max_tokens=5000,
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# --- 4. 创建 Agent ---
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

# --- 5. 运行并观察节点消息与工具决策 ---
print("🚀 [0.000s] Agent 调用发起")
# thread_id 用于标识会话；只有配置了 checkpointer 才会据此保存和恢复线程状态。
# 仅传入这个配置不会自动启用跨 invoke 的记忆。
config = {"configurable": {"thread_id": "1"}}

# 使用 stream 模式获取中间环节
# 本例未显式填写 stream_mode，按课程环境的 updates 模式理解下方节点字典。
# 这里观察的是节点完成后的消息，不是 messages 模式的逐片段输出。
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "今天什么天气？"}]},
    config=config,
    context=Context(user_id="1")
):
    # 遍历每个节点（agent 节点负责思考和决策，tools 节点负责执行）
    for node_name, output in chunk.items():
        if "messages" in output:
            last_msg = output["messages"][-1]
            
            # --- A. 尝试显示服务端显式返回的推理字段 ---
            # 是否返回此字段取决于服务端与适配器，不能保证存在。
            # additional_kwargs 是服务商额外字段；该键可能不存在，缺失时返回空字符串。
            # 本例只展示服务端已返回的信息，并不保证获得推理文本。
            reasoning = last_msg.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                print(f"\n🧠 【LLM 内部推理/Reasoning】:\n{reasoning}")

            # --- B. 打印工具调用决策 (Tool Calls) ---
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"\n🎯 【决策：调用工具】: {tc['name']} | 参数: {tc['args']}")

            # --- C. 打印最终回复内容 ---
            # 原分支把所有有正文的消息都标为“最终回复”，其中也可能包括 ToolMessage。
            # 改进展示时应按消息类型区分工具结果与 AIMessage，并检查是否仍有工具请求。
            elif last_msg.content:
                print(f"\n💬 【最终回复】: {last_msg.content}")
        

        
        # --- D. 打印工具节点返回的原始结果 ---
        # 原分支注意：tools 更新通常已包含 messages，会先进入上方 if。
        # 因此此 elif 通常不会执行，不能据此判断工具没有返回结果。
        elif node_name == "tools":
             print(f"✅ 【工具返回结果】: {output}")

timer.mark("🏁 Agent 流程结束")