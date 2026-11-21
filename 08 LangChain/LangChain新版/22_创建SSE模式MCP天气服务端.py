# 22｜创建SSE模式MCP天气服务端
# 
# 【学习目标】将天气工具通过 HTTP/SSE 暴露给 23 的客户端。
# 【阅读顺序】注册 MCP 工具 → run(transport="sse") → 保持服务运行等待客户端连接。
# 【易错提醒】先在单独终端启动本文件，再运行 23；客户端地址为 http://localhost:8000/sse。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# MCP 示例还需安装 mcp；客户端 21、23 还依赖 langchain-mcp-adapters。

from mcp.server.fastmcp import FastMCP

# 1. 初始化 FastMCP
# 本服务与 20 提供相同的模拟工具，区别在于下面使用 SSE 网络传输。
mcp = FastMCP("MyWeatherService")

# 2. 定义查天气工具
# 注册后客户端才能通过 MCP 列出和调用 query_weather；普通 Python 函数不会自动公开。
@mcp.tool()
def query_weather(city: str) -> str:
    """根据城市名称查询天气。"""
    if "北京" in city:
        return "北京天气：晴，气温 25°C，微风。"
    elif "上海" in city:
        return "上海天气：阴，气温 22°C，局部有小雨。"
    else:
        return f"暂时查不到 {city} 的天气，可能那里是世外桃源吧。"

# 只在直接运行当前脚本时执行下面的入口代码；被其他文件 import 时跳过此代码块。
if __name__ == "__main__":
    # 3. 关键修改：指定传输协议为 sse
    # 这会启动一个基于 Starlette/FastAPI 的 HTTP 服务
    # 本例使用 SSE 传输并保持服务运行；23 会连接本机的 /sse 端点。
    # SSE 是这里保留的课程示例方式，不能与另一种 Streamable HTTP 传输混为一谈。
    mcp.run(transport="sse")