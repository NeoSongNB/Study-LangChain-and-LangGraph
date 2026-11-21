# 04｜使用Pydantic提取结构化联系人信息
# 
# 【学习目标】把自然语言联系人信息提取为具有明确字段的 Pydantic 对象。
# 【阅读顺序】BaseModel 定义字段 → ToolStrategy 指定输出 → invoke → 读取 structured_response。
# 【易错提醒】ToolStrategy 依赖模型的工具调用能力；结构校验通过不代表提取的事实一定正确。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
from pprint import pprint

# 1. 加载环境变量
# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv("ARK_API_KEY")
base_url = "https://ark.cn-beijing.volces.com/api/v3"

# 2. 定义期望的输出结构 (使用 Pydantic)
# Pydantic 模型定义结构并校验字段类型；Field.description 也向模型解释字段含义。
# 这里 email 和 phone 的类型均为普通 str，不会额外验证邮箱或电话号码格式。
class ContactInfo(BaseModel):
    """提取联系人信息的结构"""
    name: str = Field(description="人员姓名")
    email: str = Field(description="电子邮箱地址")
    phone: str = Field(description="电话号码")

# 3. 初始化豆包模型
# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,  # 降低随机性；并不保证结果完全一致
    api_key=api_key,
    base_url=base_url,
)

# 4. 创建具备结构化输出能力的 Agent
# ToolStrategy 利用工具调用来约束输出结构，框架负责解析和校验。
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    system_prompt="你是一个信息提取助手，负责从文本中精准提取联系人信息。",
    # ToolStrategy 将目标结构作为工具调用格式交给模型，并在框架侧解析与校验。
    response_format=ToolStrategy(ContactInfo)
)

# 5. 调用 Agent
raw_text = "请从这里提取信息：张三，他的邮箱是 zhangsan@example.com，电话是 13800138000"
print(f"--- 正在处理文本 ---\n{raw_text}\n")

result = agent.invoke({
    "messages": [{"role": "user", "content": raw_text}]
})

# 6. 解析结果
# 核心：结构化后的数据存放在 result["structured_response"] 中
# 结构化结果是 ContactInfo 实例，可用属性访问；与 messages 中的自然语言回答分开。
structured_data = result.get("structured_response")

print("="*50)
print("【结构化提取结果】")
if structured_data:
    print(f"姓名: {structured_data.name}")
    print(f"邮箱: {structured_data.email}")
    print(f"电话: {structured_data.phone}")
    
    # 它是一个 Pydantic 对象，也可以转成字典
    # model_dump() 是 Pydantic 的实例方法，把已解析的结构化对象转为 Python 字典。
    pprint(structured_data.model_dump())
else:
    print("未能成功提取结构化数据")
