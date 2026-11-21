# 23｜Agent通过SSE会话调用MCP工具
# 
# 【学习目标】手动建立 SSE 连接和 MCP 会话，加载工具后异步调用 Agent。
# 【阅读顺序】sse_client → ClientSession → initialize → load_mcp_tools → ainvoke。
# 【易错提醒】工具调用必须发生在两个 async with 的有效期内；退出后会话和连接会关闭。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。
# MCP 示例还需安装 mcp；客户端 21、23 还依赖 langchain-mcp-adapters。

import asyncio
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 引入核心组件
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession # 必须引入这个

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()

# 先启动 22 的服务端再执行本客户端；asyncio.run 会运行这个协程。
async def main():
    # 1. 建立 SSE 连接
    # 建立网络传输通道；read/write 是 MCP 会话使用的读写流，交给 ClientSession 管理。
    async with sse_client("http://localhost:8000/sse") as (read, write):
        # 2. 核心修正：手动创建并初始化 Session
        # ClientSession 管理 MCP 请求和响应；async with 在退出作用域时释放资源。
        async with ClientSession(read, write) as session:
            # 先执行协议握手和能力协商，再列出工具；仅连上 HTTP 还不等于 MCP 会话已初始化。
            await session.initialize()
            
            # 3. 此时 load_mcp_tools 只传这一个 session 对象
            # 用已初始化的 session 加载工具；工具之后仍依赖这个会话，所以 Agent 也在作用域内执行。
            mcp_tools = await load_mcp_tools(session)

            # 4. 初始化模型
            # ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
            # model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
            # temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
            llm = ChatOpenAI(
                model="doubao-seed-1-6-251015",
                temperature=0,
                api_key=os.getenv("ARK_API_KEY"),
                base_url="https://ark.cn-beijing.volces.com/api/v3",
            )

            # 5. 创建 Agent
            # create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
            # 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
            # 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
            agent = create_agent(
                model=llm,
                system_prompt="你是智能助手小宁。请使用 query_weather 查询北京天气。",
                tools=mcp_tools,
            )

            print("--- 正在通过 HTTP (SSE) Session 查询 ---")
            response = await agent.ainvoke({
                "messages": [{"role": "user", "content": "北京天气怎么样？"}]
            })

            print(f"回答: {response['messages'][-1].content}")

# 只在直接运行当前脚本时执行下面的入口代码；被其他文件 import 时跳过此代码块。
if __name__ == "__main__":
    asyncio.run(main())