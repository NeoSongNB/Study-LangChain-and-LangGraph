# 20｜创建stdio模式MCP天气服务端
# 
# 【学习目标】把普通 Python 天气函数注册为 MCP 工具，供 21 通过子进程调用。
# 【阅读顺序】FastMCP 创建服务 → mcp.tool 注册工具 → mcp.run 启动 stdio 传输。
# 【易错提醒】天气是硬编码的模拟结果；stdio 的标准输出用于协议通信，不应随意 print 普通调试日志。
# 
# 学习注释为本次补充；原工具文档字符串和演示逻辑保留，便于与课程对照。
# 环境参考 environment.yml 中的 LangChain 1.2.3 / LangGraph 1.0.6；不要求升级到最新 API。
# MCP 示例还需安装 mcp；客户端 21、23 还依赖 langchain-mcp-adapters。

from mcp.server.fastmcp import FastMCP

# 1. 初始化 FastMCP
# FastMCP 是 MCP 服务端封装；服务名用于标识服务，不是工具函数名。
mcp = FastMCP("MyWeatherService")

# 2. 定义查天气工具
# 注册为 MCP 可发现、可调用的工具；这与 LangChain 本地的 @tool 分属不同框架。
@mcp.tool()
def query_weather(city: str) -> str:
    """根据城市名称查询天气。"""
    # 模拟数据：你可以根据输入返回不同的结果
    if "北京" in city:
        return "北京天气：晴，气温 25°C，微风。"
    elif "上海" in city:
        return "上海天气：阴，气温 22°C，局部有小雨。"
    else:
        return f"暂时查不到 {city} 的天气，可能那里是世外桃源吧。"

# 只在直接运行当前脚本时执行下面的入口代码；被其他文件 import 时跳过此代码块。
if __name__ == "__main__":
    # 默认使用 stdio，通过子进程的标准输入输出传输协议消息；日志应写到标准错误流。
    mcp.run()