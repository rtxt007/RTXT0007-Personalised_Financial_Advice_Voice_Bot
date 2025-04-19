import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import speech_recognition as sr
import pyttsx3
import tempfile
import joblib
import streamlit as st
from gtts import gTTS
import tempfile

from tensorflow.keras.models import load_model

# ========== 1. Load Pretrained Model ==========
@st.cache_resource
def load_intent_classifier(model_dir):
    model_path = os.path.join(model_dir, "model.h5")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    
    model = load_model(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, vectorizer, label_encoder

MODEL_DIR = "./financial_intent_classifier"
model, vectorizer, label_encoder = load_intent_classifier(MODEL_DIR)

# ========== [Keep all your existing helper functions] ==========
# (Keep the CRYPTO_NAME_TO_ID, COMPANY_TO_TICKER, INDICATORS, COUNTRIES,
# extract_entities, get_crypto_price, get_macro_data, fetch_financial_news,
# get_stock_data, aggregate_data, get_financial_advice, classify_intent,
# and generate_speech_file functions exactly as they are)


CRYPTO_NAME_TO_ID = {
    "Bitcoin": "bitcoin", "Ethereum": "ethereum", "Dogecoin": "dogecoin",
    "Solana": "solana", "Ripple": "ripple", "Cardano": "cardano",
    "Shiba Inu": "shiba-inu", "Litecoin": "litecoin", "Polygon": "matic-network",
    "BNB": "binancecoin"
}
COMPANY_TO_TICKER = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Tesla": "TSLA",
    "Amazon": "AMZN", "NVIDIA": "NVDA", "Tata Motors": "TATAMOTORS.NS",
    "Reliance Industries": "RELIANCE.NS", "Infosys": "INFY.NS", "Wipro": "WIPRO.NS"
}
INDICATORS = {
    "GDP": "NY.GDP.MKTP.CD", "GDP Growth Rate": "NY.GDP.MKTP.KD.ZG",
    "Inflation Rate": "FP.CPI.TOTL.ZG", "Unemployment Rate": "SL.UEM.TOTL.ZS",
    "Interest Rate": "FR.INR.RINR", "Foreign Direct Investment": "BX.KLT.DINV.WD.GD.ZS",
    "Government Debt": "GC.DOD.TOTL.GD.ZS"
}
COUNTRIES = {"USA": "US", "India": "IN", "China": "CN"}

def extract_entities(user_input):
    crypto_keywords = list(CRYPTO_NAME_TO_ID.keys())
    companies = list(COMPANY_TO_TICKER.keys())
    countries = list(COUNTRIES.keys())

    crypto_name = next((name for name in crypto_keywords if name.lower() in user_input.lower()), "Bitcoin")
    company_name = next((comp for comp in companies if comp.lower() in user_input.lower()), "Apple")
    country = next((ctry for ctry in countries if ctry.lower() in user_input.lower()), "India")

    import re
    news_keywords = re.findall(r"(?:about|regarding|on|concerning|related to)\s+([\w\s]+)", user_input, re.IGNORECASE)
    news_query = news_keywords[0].strip() if news_keywords else "stock market"

    return {
        "crypto_name": crypto_name,
        "company_name": company_name,
        "country": country,
        "news_query": news_query
    }

# ========== 3. Data Gathering ==========
def get_crypto_price(crypto_name, currency="usd"):
    crypto_id = CRYPTO_NAME_TO_ID.get(crypto_name.strip().title(), None)
    if not crypto_id:
        return {"error": f"Cryptocurrency '{crypto_name}' not found."}
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": crypto_id, "vs_currencies": currency}
    response = requests.get(url, params=params)
    return response.json()

def get_macro_data(country_code, indicator_code):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json"
    response = requests.get(url)
    data = response.json()
    return data[1][0] if data and len(data) > 1 else {"error": "No data found"}

def fetch_financial_news(query="stock market", language="en", page_size=5):
    NEWS_API_KEY = "0830d86fc57143e08d9eedc1649a9ff2"
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

def get_stock_data(company_name):
    symbol = COMPANY_TO_TICKER.get(company_name.strip().title(), None)
    if not symbol:
        return {"error": "Invalid company name."}
    stock = yf.Ticker(symbol)
    df = stock.history(period="1mo")
    df.index = df.index.strftime('%Y-%m-%d')
    return df.to_dict()

def aggregate_data(crypto_name, country, news_query, company_name):
    aggregated = {}
    aggregated["cryptocurrency"] = get_crypto_price(crypto_name, currency="usd")
    country_code = COUNTRIES.get(country.strip().title())
    macro_data = {}
    if country_code:
        for name, code in INDICATORS.items():
            macro_data[name] = get_macro_data(country_code, code)
    else:
        macro_data["error"] = "Invalid country selection"
    aggregated["macroeconomics"] = macro_data
    aggregated["financial_news"] = fetch_financial_news(query=news_query)
    aggregated["stock_market"] = get_stock_data(company_name)
    return aggregated

# ========== 4. Gemini Magic ✨ ==========
def get_financial_advice(aggregated_data):
    api_key = "AIzaSyAKxwTmHtQJE0ZxfCXIJLCNw2QJOA38HdU"
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    prompt = (
        "You are a seasoned financial advisor with deep expertise in wealth management, investment strategies, and financial planning. Using the aggregated financial data provided below, your goal is to offer actionable and well-reasoned financial advice tailored to the individual's unique financial situation. Focus on clarity, simplicity, and tangible steps that can be taken. Ensure the advice is realistic, with a brief but thorough explanation for each recommendation. Do not exceed 275 words.\n"
        f"{json.dumps(aggregated_data, indent=2)}\n\n"
    )
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(gemini_api_url, json=payload)
    response_json = response.json()
    advice = response_json['candidates'][0]['content']['parts'][0]['text']
    return advice

# ========== 5. Intent Classifier ==========
def classify_intent(user_text):
    if not user_text:
        return "Unknown"
    X = vectorizer.transform([user_text])
    pred = model.predict(X)
    pred_label = label_encoder.inverse_transform(np.argmax(pred, axis=1))
    return pred_label[0]


# ========== 6. Text-to-Speech ==========
def generate_speech_file(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts = gTTS(text=text, lang='en')
        tts.save(f.name)
        return f.name


# ========== Streamlit Interface ==========
st.set_page_config(page_title="💰 Financial AI Assistant", layout="wide")
st.title("💰 Real-Time Financial AI Assistant")
st.markdown("Type or speak your question about stocks, crypto, or macroeconomics and get personalized AI advice!")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.advice = None
    st.session_state.audio_path = None

# Input Section
with st.container():
    col1, col2 = st.columns([1, 3])
    
    with col1:
        use_speech = st.checkbox("Use Speech Input 🎤")
        audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"], 
                                    disabled=not use_speech)
        
    with col2:
        text_input = st.text_area("Or Type Your Query Here 📝", 
                                disabled=use_speech,
                                height=100)

# Process Input
if st.button("Get Financial Advice 💡"):
    input_text = ""
    
    if use_speech and audio_file is not None:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.getvalue())
            temp_path = f.name
            
        try:
            with sr.AudioFile(temp_path) as source:
                audio = recognizer.record(source)
            input_text = recognizer.recognize_google(audio)
        except Exception as e:
            st.error(f"Speech recognition error: {str(e)}")
            input_text = ""
            
    elif text_input and text_input.strip():
        input_text = text_input.strip()
        
    if input_text:
        # Process the input
        intent = classify_intent(input_text)
        entities = extract_entities(input_text)
        data = aggregate_data(entities["crypto_name"], entities["country"],
                             entities["news_query"], entities["company_name"])
        advice = get_financial_advice(data)
        audio_path = generate_speech_file(advice)
        
        # Store in session state
        st.session_state.processed = True
        st.session_state.input_text = input_text
        st.session_state.intent = intent
        st.session_state.data = data
        st.session_state.advice = advice
        st.session_state.audio_path = audio_path

# Display Results
if st.session_state.processed:
    st.divider()
    
    with st.expander("Input Analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📝 Final Input Text:**  \n`{st.session_state.input_text}`")
            st.markdown(f"**🔍 Detected Intent:**  \n`{st.session_state.intent}`")
            
        with col2:
            st.markdown("**📊 Aggregated Data**")
            st.json(st.session_state.data)
    
    st.divider()
    st.subheader("AI Financial Advice")
    st.markdown(f"💡 {st.session_state.advice}")
    
    if st.session_state.audio_path:
        st.audio(st.session_state.audio_path, format="audio/mp3")
