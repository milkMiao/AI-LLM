import os
from pathlib import Path

import dashscope
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型

# 从当前脚本所在目录加载 .env，避免受终端工作目录影响。
load_dotenv(Path(__file__).resolve().parent / ".env")

# 从环境变量获取 dashscope 的 API Key
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key:
    raise RuntimeError(
        "没有读取到 DASHSCOPE_API_KEY，请在本目录的 .env 文件中配置它。"
    )
dashscope.api_key = api_key
 
# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型

# 创建Prompt Template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# 新推荐用法：将 prompt 和 llm 组合成一个"可运行序列"
chain = prompt | llm

# 使用 invoke 方法传入输入
result1 = chain.invoke({"product": "colorful socks 哈哈哈"})
print(result1)

result2 = chain.invoke({"product": "广告设计 哈哈哈哈"})
print(result2)
