# 03｜根据用户角色生成动态提示词
# 
# 【学习目标】通过运行时 context 传递角色，让同一个 Agent 采用不同讲解风格。
# 【阅读顺序】TypedDict 声明上下文 → dynamic_prompt 读取角色 → 返回系统提示词 → invoke 传 context。
# 【易错提醒】context 是程序传入的运行配置；messages 是对话内容，两者不要混用。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
import json
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

# 1. 环境准备
# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv("ARK_API_KEY")
base_url = "https://ark.cn-beijing.volces.com/api/v3"

# 2. 定义上下文结构
# TypedDict 描述字典应包含哪些字段，便于类型检查；它本身不会自动进行运行时校验。
class Context(TypedDict):
    user_role: str

# 3. 定义动态提示词函数
# 这个装饰器会让函数拦截请求，并根据 context 修改 System Prompt
# 这个中间件每次调用模型前生成系统提示词；返回值是提示词字符串。
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:


    request_dict = vars(request)
    print(json.dumps(request_dict, indent=2, ensure_ascii=False, default=lambda x: str(x)))
    print("----------------------------------\n")

    """根据用户角色生成动态系统提示词"""
    # 💡 关键：从 request.runtime.context 获取 invoke 时传入的参数
    # 本课程使用 request.runtime.context；不要把其他路径当成可互换接口。
    # 具体访问方式需与所安装的 LangChain 版本保持一致。
    # 本示例通过 request.runtime.context 读取 invoke(context=...) 的字典。
    # 本次调用会传入非空字典；若没有传 context，需另行处理 context 为 None 的情况。
    context = getattr(request.runtime, "context", {})
    user_role = context.get("user_role", "common")
    
    base_prompt = "你是智能助手小宁。"

    if user_role == "expert":
        print("--- [动态提示词] 当前角色：专家模式 (提供深度技术回答) ---")
        return f"{base_prompt} 你现在是资深技术专家，请提供详细、深度的专业回答。"
    elif user_role == "beginner":
        print("--- [动态提示词] 当前角色：小白模式 (解释通俗易懂) ---")
        return f"{base_prompt} 你现在是启蒙老师，请用最通俗易懂、大白话的方式解释概念。"
    
    return base_prompt

# 4. 初始化豆包模型
# ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
# model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
# temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

# 5. 创建 Agent (将动态提示词作为中间件传入)
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=llm,
    middleware=[user_role_prompt], # 👈 这里注入动态逻辑
    # context_schema 声明运行时上下文结构；实际数据仍需在 invoke/stream 的 context 参数中传入。
    context_schema=Context         # 👈 定义 context 的类型约束
)

# 6. 运行案例：以“专家”身份询问
print("\n--- 第一次测试：专家模式 ---")
# 给同一个 Agent 传不同角色上下文，就能切换回答风格；下方注释掉的 beginner 示例可作对照。
result_expert = agent.invoke(
    {"messages": [{"role": "user", "content": "你是谁"}]},
    context={"user_role": "expert"} # 👈 传入 context
)
print(f"专家回答: {result_expert['messages'][-1].content}")

# 7. 运行案例：以“小白”身份询问
# print("\n--- 第二次测试：小白模式 ---")
# result_beginner = agent.invoke(
#     {"messages": [{"role": "user", "content": "简单解释一下什么是大语言模型"}]},
#     context={"user_role": "beginner"} # 👈 传入 context
# )
# print(f"小白回答: {result_beginner['messages'][-1].content}")