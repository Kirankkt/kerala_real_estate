import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
#from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# Title and Navigation
st.title('Plot Price Predictor and CrewAI Assistant')
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "View Analytics"])
# API Key Input
#api_key = st.text_input('Please enter your OpenAI API key:', type='password')

# Load the model and scaler
model = joblib.load('gboost_model_9.pkl')
scaler = joblib.load('scaler.pkl')

# Load the DataFrame
df = pd.read_csv('Hplot_df_filer_cleaned_no_outliers.csv')

#if api_key:
    # Set OpenAI API key
    #os.environ["OPENAI_API_KEY"] = api_key

    # Initialize the language model
    #llm = ChatOpenAI(temperature=0.7)

    # Initialize a conversational memory for the LLM
    #memory = ConversationBufferMemory()

    # Create a Pandas DataFrame agent
    #dataframe_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# Main Predictor Page
if page == "Predictor":
    # Mapping locations to their corresponding numerical values
    location_mapping = {
        'akg nagar': 0,
        'alamcode': 1,
        'anayara': 2,
        'andoorkonam': 3,
        'anugraha': 4,
        'aradhana nagar': 5,
        'attingal': 6,
        'azhikode': 7,
        'chanthavila': 8,
        'chempazhanthy': 9,
        'chenkottukonam': 10,
        'chirayinkeezhu': 11,
        'chittattumukku': 12,
        'dhanuvachapuram': 13,
        'gandhipuram': 14,
        'gayatri': 15,
        'kaimanam': 16,
        'kakkamoola': 17,
        'kallambalam': 18,
        'kallayam': 19,
        'kalliyoor': 20,
        'kaniyapuram': 21,
        'kanjiramkulam': 22,
        'karakulam': 23,
        'karamana': 24,
        'kariavattom': 25,
        'karikkakam': 26,
        'karimanal': 27,
        'karyavattom': 28,
        'kattaikonam': 29,
        'kattakada': 30,
        'kattakkada': 31,
        'kazhakootam': 32,
        'kesavadasapuram': 33,
        'kilimanoor': 34,
        'killy': 35,
        'kollamkavu': 36,
        'kovalam': 37,
        'kowdiar': 38,
        'kulathoor': 39,
        'kunnapuzha': 40,
        'kuravankonam': 41,
        'malayinkeezhu': 42,
        'mangalapuram': 43,
        'manikanteswaram': 44,
        'mannanthala': 45,
        'maruthankuzhi': 46,
        'maruthoor': 47,
        'menamkulam': 48,
        'moongod': 49,
        'mudavanmugal': 50,
        'mukkola': 51,
        'muttathara': 52,
        'nalanchira': 53,
        'nemom': 54,
        'nettayam': 55,
        'njandoorkonam': 56,
        'ooruttambalam': 57,
        'pachalloor': 58,
        'palkulangara': 59,
        'pallichal': 60,
        'pappanamcode': 61,
        'parottukonam': 62,
        'paruthippara': 63,
        'pattom': 64,
        'peringammala': 65,
        'peroorkada': 66,
        'pettah': 67,
        'peyad': 68,
        'pidaram': 69,
        'poojappura': 70,
        'pothencode': 71,
        'pottayil': 72,
        'powdikonam': 73,
        'pravachambalam': 74,
        'premier sarayu': 75,
        'puliyarakonam': 76,
        'punnakkamughal': 77,
        'puthenthope': 78,
        'shangumukham': 79,
        'sreekariyam': 80,
        'surabhi gardens': 81,
        'thachottukavu': 82,
        'thattathumala': 83,
        'thirumala': 84,
        'udiyankulangara': 85,
        'uliyazhathura': 86,
        'ulloor': 87,
        'vattapara': 88,
        'vattiyoorkavu': 89,
        'vazhayila': 90,
        'vazhuthacaud': 91,
        'vellayani': 92,
        'venjaramoodu': 93,
        'vettamukku': 94,
        'vizhinjam': 95
    }


    # Location Dropdown
    selected_location = st.selectbox('Select Location', list(location_mapping.keys()))

    # Convert the selected location to its numerical value
    location = location_mapping[selected_location]

    # User inputs
    property_age = st.number_input('Property Age')
    bedroom_count = st.number_input('Bedroom Count')
    # User inputs in cents
    build_area = st.number_input('Build Area ')
    plot_area_cents = st.number_input('Plot Area (in cents)')
    
    # Convert cents to square feet
    # 1 cent = 435.6 sqft
    #build_area = build_area_cents * 435.6
    plot_area = plot_area_cents * 435.6

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



# Analytics Dashboard
elif page == "View Analytics":
    st.header("Analytics Dashboard")

    # Display downloaded charts (assuming PNG or JPEG)
    st.subheader("Correlation_chart")
    st.image('corr_chat.png', use_column_width=True)

    st.subheader("Average plot area by location")
    st.image('avg_plot_area_by_location.png', use_column_width=True)

    st.subheader("Number of plots by location")
    st.image('plots_by_location.png', use_column_width=True)

    st.subheader("Average price by location")
    st.image('avg_price_by_location.png', use_column_width=True)

    st.subheader("Exploratory Data Analysis (EDA) Report")

    # Path to the EDA report HTML file
    with open('eda_report (1).html', 'r', encoding='utf-8') as f:
        eda_html = f.read()
    
    # Display the HTML content with increased height
    components.html(eda_html, height=1000, scrolling=True)  # Increased height to 1000



















