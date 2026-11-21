# 16｜使用内存检查点保存并查看会话状态
# 
# 【学习目标】启用 InMemorySaver 保存线程状态，并读取消息历史与快照元数据。
# 【阅读顺序】创建 checkpointer → 传 thread_id → 调用 Agent → get_state(config) 查看快照。
# 【易错提醒】内存检查点仅在当前进程内保存；关闭进程后不能像数据库一样恢复记录。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
from pprint import pprint
import time
import json
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  
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
        print(f"⏱️ [{elapsed_total:.3f}s] {event_name} (距离上一阶段: {since_last:.3f}s)")
        self.last_mark = now

timer = AgentTimer()

# --- 1. 系统提示词 (简化了逻辑，不再要求先调位置工具) ---
SYSTEM_PROMPT = "你是一位专业的专业气象预报员。请使用 get_weather_info 获取天气信息。如果你不知道用户在哪，工具会自动处理。"

# dataclass 自动生成初始化方法，因此可以写 Context(user_id="1")。
# 这个上下文由应用提供，适合放用户身份等依赖；它不等同于 messages 对话历史。
@dataclass
class Context:
    user_id: str

# --- 2. 合并后的超级工具 ---
# 把确定的业务流程封装在一个工具内，模型只需要决定是否调用以及填写 city。
# runtime 由框架注入；city 缺省时，下面的 Python 代码会自己确定用户位置。
# 原注解 city: str = None 保留；若改进类型表达，可使用 city: str | None = None。
@tool
def get_weather_info(runtime: ToolRuntime[Context], city: str = None) -> str:
    """获取天气信息。如果 city 为空，将自动根据当前用户 ID 识别位置。"""
    timer.mark(f"➡️ 超级工具 [get_weather_info] 开始执行 (输入城市: {city})")
    
    # 逻辑 A：如果模型没给城市，内部执行位置识别
    target_city = city
    if not target_city:
        user_id = runtime.context.user_id
        # 模拟内部位置查询逻辑
        target_city = "佛罗里达" if user_id == "1" else "旧金山"
        print(f"  [工具内部日志] 自动识别位置为: {target_city}")
    
    # 逻辑 B：执行天气查询
    result = f"{target_city} 总是阳光明媚！"
    
    return result

# --- 3. 初始化 LLM ---
# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=0,
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# --- 4. 创建 Agent (只注册一个工具) ---
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    # 注册可供模型选择的工具；列表顺序不保证执行顺序，具体调用由模型和业务逻辑决定。
    tools=[get_weather_info],
    # context_schema 声明运行时上下文结构；实际数据仍需在 invoke/stream 的 context 参数中传入。
    context_schema=Context,
    # 真正启用线程状态保存的是这里的 checkpointer，不是单独设置 thread_id。
    checkpointer=InMemorySaver(), 
)

# --- 5. 运行并分析流程耗时 ---
print("🚀 [0.000s] Agent 调用发起 (已合并工具)")
# thread_id 用于标识会话；只有配置了 checkpointer 才会据此保存和恢复线程状态。
# 仅传入这个配置不会自动启用跨 invoke 的记忆。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "今天什么天气？"}]},
    config=config,
    context=Context(user_id="1")
)

timer.mark("🏁 Agent 流程结束，返回最终答案")

print("\n----------------------------------")
print(f"最终回答: {response['messages'][-1].content}")

# --- 6. 查看 Checkpointer 中的存储内容 ---
print("\n🔍 [检查点内容查询] ...")

# 使用 config 获取当前 thread_id 的最新状态
# 用同一个 thread_id 获取最新 StateSnapshot；get_state 读取状态，不会再请求模型。
state = agent.get_state(config)



print("="*50)
print(f"线程 ID: {config['configurable']['thread_id']}")
# state.next 表示接下来待执行的节点，而不是刚执行过的节点；完成后通常是空元组。
print(f"当前节点: {state.next}") # 接下来要运行的节点（如果是结束了，这里通常为空）

print("\n--- 存储的消息历史 (Messages) ---")
# state.values 是业务状态，messages 包含用户、模型和工具的完整消息链。
for i, msg in enumerate(state.values["messages"]):
    # 打印消息类型、ID和内容片段
    print(f"[{i}] {msg.__class__.__name__} (ID: {msg.id}): {msg.content}")
    
    # 如果消息包含工具调用，也打印出来
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"    └─ 🛠️ 包含工具指令: {msg.tool_calls[0]['name']}")

print("\n--- 原始快照元数据 (Metadata) ---")
# 打印当前快照 metadata，具体字段取决于版本；创建时间等信息不一定在这里。
# metadata 是检查点管理元信息；具体键和值由框架和版本决定，不能固定假设一定含时间戳。
print(json.dumps(state.metadata, indent=2, ensure_ascii=False))
print("="*50)

pprint(state.values)