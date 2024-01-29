from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

df = pd.read_csv('calorie.csv')

# from langchain.llms import OpenAI
lastPrompt = "画像に文字を使用する際は、すべての文字のフォントを「Meiryo」にしてください。"
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

agent.run("料理の種類ごとのエネルギーの総和を棒グラフ画像形式で、graphsフォルダ内にエクスポートしてください。" + lastPrompt)