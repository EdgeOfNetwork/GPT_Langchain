import pandas as pd
from langchain import OpenAI
import os
import config

df = pd.read_csv("https://github.com/kairess/toy-datasets/raw/master/titanic.csv")

from langchain.agents import create_pandas_dataframe_agent


os.environ["OPENAI_API_KEY"] = config.api_key
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

print(agent.run("how many rows are there?"))
print(agent.run("행이 몇 개지?"))
print(agent.run("승객들의 평균 연령은?"))
print(agent.run("남성과 여성의 비율은?"))
print(agent.run("객실 등급과 성별에 따른 생존자 수를 계산해줘"))