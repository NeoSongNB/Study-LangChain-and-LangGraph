# 21｜Agent通过stdio调用MCP天气工具
# 
# 【学习目标】用 MCP 适配器启动本地服务子进程，并把远程工具交给 Agent。
# 【阅读顺序】配置 stdio → get_tools 发现工具 → create_agent → await ainvoke。
# 【易错提醒】服务端由客户端按配置启动，无需先手动运行 20；已同步适配重命名后的文件路径。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# 运行前：准备依赖和包含 ARK_API_KEY 的 .env；模型名称沿用原代码，需账号实际可调用。
# MCP 示例还需安装 mcp；客户端 21、23 还依赖 langchain-mcp-adapters。

import os
import asyncio
import sys
from pathlib import Path
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

# 读取 .env，将其中的配置加载到环境变量；默认不会覆盖已经存在的环境变量。
# 本例从 ARK_API_KEY 读取密钥，示例不把密钥直接写进代码。
load_dotenv()

# async 定义协程；await 等待网络/工具等异步操作，asyncio.run 在入口驱动事件循环。
async def main():
    # 一行代码替代所有的 stdio/ClientSession 管理
    # weather 是此客户端配置中的服务别名；stdio 表示由客户端启动本地 Python 子进程。
    # command 指定解释器，args 指定服务端脚本；适配器负责会话和工具转换。
    client = MultiServerMCPClient({
        "weather": {
            "transport": "stdio",
            "command": sys.executable,  # 使用当前环境的解释器启动服务端。
            # 以脚本所在目录定位服务端，避免从项目根目录运行时相对路径失效。
            "args": [str(Path(__file__).resolve().with_name("20_创建stdio模式MCP天气服务端.py"))],
            "env": os.environ.copy()
        }
    })

    # 获取工具
    # 通过 MCP 发现服务端工具，并转换为 LangChain Agent 可以使用的工具对象。
    tools = await client.get_tools()

    # 初始化模型
    # ChatOpenAI 是聊天模型客户端；这里使用兼容 OpenAI 协议的豆包服务地址。
    # model 指定服务端模型，api_key 用于认证，base_url 指向实际服务商。
    # temperature 控制采样随机性；设为 0 通常更稳定，但不保证每次回答完全相同。
    llm = ChatOpenAI(
        model="doubao-seed-1-6-251015",
        temperature=0,
        api_key=os.getenv("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    # 创建 Agent（一句话）
    # create_agent 将模型、工具及可选中间件组装成可执行的 Agent 图。
    # 有工具时通常循环执行：模型决定工具与参数 → Python 执行工具 → 结果回传模型。
    # 模型输出 tool_calls 并不等于工具已执行，真正的函数调用由框架负责。
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是智能助手小宁。请使用 query_weather 查询北京天气。",
    )

    # 执行
    # 异步 ainvoke 对应同步 invoke；仍然要等 Agent 执行完成才返回最终状态。
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "北京天气怎么样？"}]
    })

    print(f"回答: {response['messages'][-1].content}")

# 只在直接运行当前脚本时执行下面的入口代码；被其他文件 import 时跳过此代码块。
if __name__ == "__main__":
    asyncio.run(main())