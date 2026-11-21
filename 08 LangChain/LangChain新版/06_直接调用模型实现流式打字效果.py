# 06｜直接调用模型实现流式打字效果
# 
# 【学习目标】绕过 Agent，直接用聊天模型的 stream 逐片段输出文本。
# 【阅读顺序】组装 system/user 消息 → llm.stream → 读取 chunk.content → 连续打印。
# 【易错提醒】片段不保证恰好一个字或一个 token；end 与 flush 配合产生逐步显示的效果。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import json
import os
from langchain_openai import ChatOpenAI
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
    model="doubao-seed-1-6-251015",
    temperature=0,
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

print("LLM streaming (typing effect):")
# Direct LLM stream for token-by-token
# 直接调用模型时，把系统角色和用户问题一起放进列表；这里不经过 Agent 工具循环。
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "罗永浩帅吗？"}
]
# llm.stream 返回消息片段迭代器；消费迭代器时才会持续接收输出。
for chunk in llm.stream(messages):


    # print("----------------------------------\n")
    # print(json.dumps(chunk, indent=2, ensure_ascii=False, default=lambda x: str(x)))

    # 结束、用量等片段可能没有文本，因此先检查 content。
    # end="" 避免每片段换行；flush=True 及时刷新终端输出。
    if chunk.content:
        print(chunk.content, end='', flush=True)

# 原脚本到这里仅打印一个标题，没有执行额外的 Agent 调用或非流式请求。
print("\n\nAgent full response (non-streaming):")
