import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import speech_recognition as sr
import tempfile
import joblib
import streamlit as st
from gtts import gTTS
from tensorflow.keras.models import load_model

# ——— STREAMLIT CONFIGURATION ———
# Must be called once and before any other Streamlit commands
st.set_page_config(
    page_title="💰 Financial AI Assistant",
    layout="wide"
)

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

# ========== Entity Constants and Functions ==========
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

# ========== Data Gathering Functions ==========
def get_crypto_price(crypto_name, currency="usd"):
    crypto_id = CRYPTO_NAME_TO_ID.get(crypto_name.strip().title())
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
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
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
    symbol = COMPANY_TO_TICKER.get(company_name.strip().title())
    if not symbol:
        return {"error": "Invalid company name."}
    stock = yf.Ticker(symbol)
    df = stock.history(period="1mo")
    df.index = df.index.strftime('%Y-%m-%d')
    return df.to_dict()

def aggregate_data(crypto_name, country, news_query, company_name):
    return {
        "cryptocurrency": get_crypto_price(crypto_name),
        "macroeconomics": get_macro_data(COUNTRIES.get(country.strip().title()), INDICATORS),
        "financial_news": fetch_financial_news(news_query),
        "stock_market": get_stock_data(company_name)
    }

# ========== AI Advice Generation ==========
def get_financial_advice(aggregated_data):
    api_key = st.secrets["GEMINI_API_KEY"]
    gemini_api_url = (
        f"https://generativelanguage.googleapis.com"
        f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    )

    prompt = (
        "You are a seasoned financial advisor with deep expertise in wealth management.\n"
        f"Provide actionable advice based on this data: {json.dumps(aggregated_data, indent=2)}\n"
        "Focus on clarity and tangible steps. Keep under 275 words."
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}  
        ]
    }

    response = requests.post(gemini_api_url, json=payload)
    response.raise_for_status()
    candidates = response.json().get("candidates", [])
    if not candidates:
        raise ValueError("No response candidates returned from Gemini API.")
    return candidates[0]["content"]["parts"][0]["text"]

# ========== Core Application Logic ==========
def classify_intent(user_text):
    if not user_text:
        return "Unknown"
    X = vectorizer.transform([user_text])
    pred = model.predict(X)
    return label_encoder.inverse_transform(np.argmax(pred, axis=1))[0]

def generate_speech_file(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(f.name)
        return f.name

# ========== Streamlit UI ==========
st.title("💰 Real-Time Financial AI Assistant")
st.markdown("Type or speak your question about stocks, crypto, or macroeconomics")

if 'processed' not in st.session_state:
    st.session_state.update({
        'processed': False,
        'input_text': '',
        'intent': '',
        'data': {},
        'advice': '',
        'audio_path': None
    })

with st.container():
    use_speech = st.checkbox("Use Speech Input 🎤")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"], disabled=not use_speech)
    text_input = st.text_area("Or type your query", height=100, disabled=use_speech)

if st.button("Get Financial Advice 💡"):
    input_text = ""
    if use_speech and audio_file:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_file.getvalue())
            try:
                with sr.AudioFile(tmp.name) as source:
                    audio = sr.Recognizer().record(source)
                    input_text = sr.Recognizer().recognize_google(audio)
            except Exception as e:
                st.error(f"Speech recognition error: {str(e)}")
    elif text_input:
        input_text = text_input.strip()

    if input_text:
        intent = classify_intent(input_text)
        entities = extract_entities(input_text)
        data = aggregate_data(**entities)
        advice = get_financial_advice(data)
        st.session_state.update({
            'processed': True,
            'input_text': input_text,
            'intent': intent,
            'data': data,
            'advice': advice,
            'audio_path': generate_speech_file(advice)
        })

if st.session_state.processed:
    st.divider()
    with st.expander("Analysis", expanded=True):
        cols = st.columns(2)
        cols[0].write(f"**Input:**\n{st.session_state.input_text}")
        cols[0].write(f"**Intent:**\n{st.session_state.intent}")
        cols[1].json(st.session_state.data)
    st.divider()
    st.subheader("AI Financial Advice")
    st.write(st.session_state.advice)
    if st.session_state.audio_path:
        st.audio(st.session_state.audio_path, format="audio/mp3")
