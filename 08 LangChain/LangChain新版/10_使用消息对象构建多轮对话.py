# 10｜使用消息对象构建多轮对话
# 
# 【学习目标】用 SystemMessage、HumanMessage、AIMessage 手动组织聊天历史。
# 【阅读顺序】构造不同角色的消息对象 → 按时间排序 → llm.invoke → 观察 AIMessage。
# 【易错提醒】这里的历史由列表显式传入；模型不会自动记住另一次独立请求。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage  # 👈 langchain_core
from pprint import pprint
from dotenv import load_dotenv

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv("ARK_API_KEY")

SYSTEM_PROMPT = "你是智能助手小宁"

# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-8-251228",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 对话历史（BaseMessage 对象 ✅）
# 手工构造的 AIMessage 是示例历史，并非当前模型刚生成的结果。
ai_msg = AIMessage(content="I'd be happy to help you with that question!")



# SystemMessage 指定助手身份；HumanMessage 表示用户；AIMessage 表示助手历史回复。
# 列表顺序就是模型看到的对话顺序。
messages = [
    SystemMessage(content=SYSTEM_PROMPT),  # 👈 用 SYSTEM_PROMPT
    HumanMessage(content="Can you help me?"),
    ai_msg,
    HumanMessage(content="Great! What's 2+2?"),
]

# pprint(messages)

# print("="*50)
# print(json.dumps(messages, indent=2, ensure_ascii=False, default=lambda x: str(x)))

# print("Messages:")
for msg in messages:
    print(f"{msg.__class__.__name__}: {msg.content}")

# 👈 直接 llm.invoke(messages) – 官方支持！
# 直接 llm.invoke 返回 AIMessage 对象；与 agent.invoke 返回状态字典不同。
response = llm.invoke(messages)


print("="*50)
print(json.dumps(response, indent=2, ensure_ascii=False, default=lambda x: str(x)))