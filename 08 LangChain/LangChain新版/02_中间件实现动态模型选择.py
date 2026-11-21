# 02｜中间件实现动态模型选择
# 
# 【学习目标】在一次模型调用前，按消息条数选择基础模型或高级模型。
# 【阅读顺序】准备两个模型 → wrap_model_call 拦截 → override 替换模型 → handler 继续执行。
# 【易错提醒】阈值是大于 6 条消息，不是 6 轮对话；当前示例只有 5 条消息，会选择基础模型。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
import json



# 加载环境变量
# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()
api_key = os.getenv("ARK_API_KEY")
base_url = "https://ark.cn-beijing.volces.com/api/v3"

# 定义模型
basic_model = ChatOpenAI(
    model="doubao-seed-1-6-lite-251015",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

advanced_model = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0.5,
    api_key=api_key,
    base_url=base_url,
)


# 中间件 - 关键修复点
# 环绕中间件类似 Java 的拦截器：拿到请求，做前置逻辑，再交给 handler 继续处理。
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:

    # vars 只用于调试观察请求属性；不要依赖整个内部结构作为稳定的业务接口。
    request_dict = vars(request)
    print(json.dumps(request_dict, indent=2, ensure_ascii=False, default=lambda x: str(x)))
    print("----------------------------------\n")

    """动态模型选择"""
    # ✅ 修复1: 用 request.messages，不是 request.state.get()
    # 这里只按当前请求的消息条数选模型；一轮对话可能包含多条消息和工具结果。
    messages = request.messages
    message_count = len(messages)

    print(f"\n[中间件] 消息数: {message_count}")

    if message_count > 6:
        print("🚀 高级模型")
        model = advanced_model
    else:
        print("☁️ 基础模型")
        model = basic_model

    # ✅ 修复2: request.override(model=model)
    # override 返回替换模型后的请求；必须调用 handler 才会实际执行后续模型调用。
    # 返回 handler 的结果，才能让 Agent 继续收到模型响应。
    return handler(request.override(model=model))


# 创建 agent - 无工具纯聊天
# create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
# 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
# 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
agent = create_agent(
    model=basic_model,
    system_prompt="你是智能助手小宁",  # ✅ v1 用 system_prompt
    middleware=[dynamic_model_selection],  # ✅ 无 tools= 也能跑
)

# 运行 - ✅ 输入输出格式
# 只在直接运行当前脚本时执行下面的入口代码；被其他文件 import 时跳过此代码块。
if __name__ == "__main__":
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，我是小宁。"},
                {"role": "user", "content": "我想写 Agent 中间件"},
                {"role": "assistant", "content": "可以用 Python 实现。"},
                {"role": "user", "content": "你好帅啊"},
            ]
        }
    )

    print("\n" + "=" * 50)
    print("✅ 最终响应:")
    print(response["messages"][-1].content)  # ✅ 正确：AIMessage.content
    print("=" * 50)
