# 01｜创建基础Agent与查看响应
# 
# 【学习目标】用 ChatOpenAI 接入豆包，创建无工具 Agent，并观察返回的消息状态。
# 【阅读顺序】加载配置 → 创建模型 → create_agent → invoke → 对比三种打印方式。
# 【易错提醒】invoke 返回包含 messages 的状态字典；最后一条 AIMessage 的 content 才通常是回答正文。
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


# 这里再次读取同一个变量：os.environ[...] 在缺失时会抛 KeyError；os.getenv 则默认返回 None。
api_key = os.environ["ARK_API_KEY"]
# 将已读取的密钥传给客户端，无需设置 OPENAI_API_KEY。
# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=api_key,  # 👈 直接传入API Key
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 👈 直接传入Base URL
)

# Define response format



# Create agent
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
)

# invoke 是同步调用，会等本次 Agent 执行结束后才返回状态字典。
# messages 中 role="user" 表示用户输入；system_prompt 在创建 Agent 时单独指定。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "你是谁？"}]},
)

print(response)
print("="*50)
pprint(response)
print("="*50)
# 用 JSON 排版输出状态；不可序列化的消息对象由 default 转成字符串。
# default=lambda x: str(x) 把无法 JSON 编码的对象转为字符串，便于调试。
# 它并不是把消息对象逐字段转为字典；需要结构化导出时可单独对消息调用 model_dump()。
print(json.dumps(response, indent=2, ensure_ascii=False, default=lambda x: str(x)))




