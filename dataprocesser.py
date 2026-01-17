import pandas as pd
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Set your API Key (Starts with nvapi-)
os.environ["NVIDIA_API_KEY"] = "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"

# 2. Initialize the NVIDIA Model
# Llama-3.1-70b or 405b are excellent for complex reasoning
llm = ChatNVIDIA(model="deepseek-ai/deepseek-v3.2", temperature=1)

# 3. Load your data
df = pd.read_csv("sample.csv")

# 4. Create the Agent
agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True, 
    allow_dangerous_code=True # Required in newer LangChain versions
)
print(df.columns.tolist()) 
print(df.shape) 

try: 
    resp1 = agent.invoke("How many rows are in the dataframe? Reply with just the number.") 
    print("Agent rows response:", resp1) 
    resp2 = agent.invoke("What are the column names? Reply as a Python list.") 
    print("Agent columns response:", resp2) 
    resp3 = agent.invoke("Show me the first 3 rows.") 
    print("Agent head response:", resp3) 
except Exception as e: 
    print("Agent error:", e)