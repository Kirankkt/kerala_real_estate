#from langchain import LangChainAgent, Memory
#from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
import pandas as pd
import joblib
import os


# Load the data and model
df = pd.read_csv(r'C:\Users\kthom\Documents\Kerala_Hosuing_RU\Hplot_df_filer_cleaned_no_outliers.csv')
model = joblib.load('xgboost_model_7.pkl')
# Create a conversational memory
memory = ConversationBufferMemory()


# Set up OpenAI LLM
# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"
# Initialize the language model
llm = ChatOpenAI(temperature=0.7)

# Create an agent executor with conversational memory
agent_executor = create_pandas_dataframe_agent(llm, df, memory=memory, verbose=True, allow_dangerous_code=True)


# Optionally test LangChain setup
print("LangChain setup complete.")
