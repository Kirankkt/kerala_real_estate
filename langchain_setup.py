import pandas as pd
import joblib
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

# Load the data and model
df = pd.read_csv('Hplot_df_filer_cleaned_no_outliers.csv')
model = joblib.load('gboost_model_7.pkl')

# Set OpenAI API Key (for testing or setup; in practice, handle securely in app.py)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is not set. Please set the environment variable 'OPENAI_API_KEY'.")

# Initialize the language model
llm = ChatOpenAI(temperature=0.7)

# Create a conversational memory
memory = ConversationBufferMemory()

# Create an agent executor with conversational memory
agent_executor = create_pandas_dataframe_agent(llm, df, memory=memory, verbose=True, allow_dangerous_code=True)

# Optionally test LangChain setup
print("LangChain setup complete.")
