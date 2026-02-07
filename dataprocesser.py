import pandas as pd
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_experimental.agents import create_pandas_dataframe_agent
def pandas_agent():

    # 1. Set your API Key (Starts with nvapi-)
    os.environ["NVIDIA_API_KEY"] = "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"

# 2. Initialize the NVIDIA Model
# deepseek-ai/deepseek-v3.2 are excellent for complex reasoning
    llm = ChatNVIDIA(model="deepseek-ai/deepseek-v3.2", temperature=0.2, request_timeout=120)

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
        #resp1 = agent.invoke(input("enter question: ")) 
        #print("Agent rows response:", resp1) 
        resp2 = agent.invoke("How many rows?") 
        print("Agent columns response:", resp2) 
        resp3 = agent.invoke("what is the mean value of the 'Revenue' column?") 
        print("Agent head response:", resp3) 
    except Exception as e: 
        print("Agent error:", e)

if __name__ == "__main__":
    pandas_agent()
