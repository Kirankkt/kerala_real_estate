import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load the model and scaler
model = joblib.load('gboost_model_7.pkl')
scaler = joblib.load('scaler.pkl')

# Verify that the scaler is loaded correctly
#st.write("Scaler mean:", scaler.mean_)
#st.write("Scaler scale:", scaler.scale_)



# Load the DataFrame
df = pd.read_csv(r'C:\Users\kthom\Documents\Kerala_Hosuing_RU\Hplot_df_filer_cleaned_no_outliers.csv')

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = input("Please enter your OpenAI API key: ")

# Initialize the language model
llm = ChatOpenAI(temperature=0.7)

# Initialize a conversational memory for the LLM
memory = ConversationBufferMemory()

# Create a LangChain conversation chain
conversation = ConversationChain(llm=llm, memory=memory)

# Create a Pandas DataFrame agent
dataframe_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

# Streamlit UI
st.title('Plot Price Predictor and Data Chat')

# User inputs
location = st.number_input('Location')
property_age = st.number_input('Property Age')
bedroom_count = st.number_input('Bedroom Count')
build_area = st.number_input('Build Area')
plot_area = st.number_input('Plot Area')

# Prepare input for prediction
input_data = pd.DataFrame({
    'Plot__Beds': [bedroom_count],
    'Property__Age': [property_age],
    'Build__Area': [build_area],
    'Plot__Area': [plot_area],
    'Plot__Location': [location]
})

# Separate numerical columns for scaling
numerical_columns = ['Plot__Beds', 'Property__Age', 'Build__Area', 'Plot__Area']

# Transform the data
input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

# Predict price
if st.button('Predict Price'):
    predicted_price = model.predict(input_data)
    st.write(f'Predicted Plot Price: {predicted_price[0]}')

# Chat with the DataFrame using LLM
user_query = st.text_input('Ask a question about the data:')
if st.button('Get Answer'):
    response = dataframe_agent.run(user_query)
    st.write(response)
