import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent
import dashscope

# 从当前脚本所在目录加载 .env，确保 SerpAPI 工具也能读取 Key。
load_dotenv(Path(__file__).resolve().parent / ".env")

# 从环境变量获取 dashscope 的 API Key
api_key = os.getenv('DASHSCOPE_API_KEY')
serpapi_key = os.getenv('SERPAPI_API_KEY')
if not api_key:
    raise RuntimeError("没有读取到 DASHSCOPE_API_KEY，请检查本目录的 .env 文件。")
if not serpapi_key:
    raise RuntimeError("没有读取到 SERPAPI_API_KEY，请检查本目录的 .env 文件。")
dashscope.api_key = api_key

# 加载模型 (使用 ChatModel 以支持 tool calling)
llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

# 加载 serpapi 工具
tools = load_tools(["serpapi"])

# LangChain 1.x 新写法
agent = create_agent(llm, tools)

# 运行 agent
result = agent.invoke({"messages": [("user", "今天是几月几号?历史上的今天有哪些名人出生")]})
print(result["messages"][-1].content)
