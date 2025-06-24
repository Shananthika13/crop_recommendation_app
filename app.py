import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Enhanced page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with premium styling
st.markdown(r"""
<style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap");
    
    /* Root variables for consistent theming */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --dark-bg: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        --card-bg: rgba(255, 255, 255, 0.08);
        --border-color: rgba(255, 255, 255, 0.15);
        --text-primary: #ffffff;
        --text-secondary: #b8c2cc;
        --accent-blue: #64b5f6;
        --accent-green: #81c784;
        --accent-purple: #ba68c8;
        --accent-orange: #ffb74d;
        --shadow-primary: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        background: var(--dark-bg);
        min-height: 100vh;
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
    }
    
    /* Enhanced animations */
    @keyframes pulse-glow {
        0% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
            transform: scale(1);
        }
        50% {
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.6);
            transform: scale(1.02);
        }
        100% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
            transform: scale(1);
        }
    }
    
    @keyframes slide-in {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: var(--primary-gradient);
        background-size: 400% 400%;
        animation: gradient-shift 8s ease infinite;
        margin: -2rem -1rem 2rem -1rem;
        border-radius: 0 0 30px 30px;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        color: var(--text-primary);
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 20px rgba(0, 0, 0, 0.5);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        font-weight: 400;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced card styling */
    .premium-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-primary);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slide-in 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: var(--primary-gradient);
        border-radius: 20px 20px 0 0;
    }
    
    .premium-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-hover);
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        text-align: center;
        padding: 2rem 1rem;
        background: var(--card-bg);
        border-radius: 16px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .feature-item:hover::before {
        left: 100%;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        border-color: var(--accent-blue);
        box-shadow: 0 10px 30px rgba(100, 181, 246, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        background: var(--primary-gradient) !important;
        color: var(--text-primary) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-primary) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: var(--shadow-hover) !important;
        background: var(--secondary-gradient) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Enhanced slider styling */
    .stSlider {
        padding: 1.5rem 0;
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary-gradient) !important;
        border-radius: 10px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: var(--text-primary) !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        border: 3px solid var(--accent-blue) !important;
    }
    
    /* Enhanced input styling */
    .stNumberInput > div > div > input {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 20px rgba(100, 181, 246, 0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: var(--card-bg);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        border: 1px solid transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--accent-blue) !important;
        box-shadow: 0 4px 16px rgba(100, 181, 246, 0.3) !important;
    }
    
    /* Result card styling */
    .result-card {
        background: var(--success-gradient);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: var(--shadow-primary);
        animation: pulse-glow 3s ease-in-out infinite;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: gradient-shift 4s ease infinite;
    }
    
    .result-card h1 {
        font-size: 4rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        text-shadow: 2px 2px 20px rgba(0, 0, 0, 0.3) !important;
        position: relative;
        z-index: 1;
    }
    
    .result-card h2 {
        font-size: 1.5rem !important;
        color: rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 1rem !important;
        position: relative;
        z-index: 1;
    }
    
    /* Detail cards */
    .detail-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .detail-item {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .detail-item:nth-child(1) { border-top: 3px solid var(--accent-green); }
    .detail-item:nth-child(2) { border-top: 3px solid var(--accent-blue); }
    .detail-item:nth-child(3) { border-top: 3px solid var(--accent-purple); }
    .detail-item:nth-child(4) { border-top: 3px solid var(--accent-orange); }
    
    .detail-item:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-hover);
    }
    
    .detail-item h4 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .detail-item p {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin: 0;
    }
    
    /* Section headers */
    .section-header {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-align: center;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: var(--primary-gradient);
        border-radius: 2px;
    }
    
    /* Parameter cards */
    .param-section {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .param-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: var(--primary-gradient);
    }
    
    /* Chart container */
    .chart-container {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-primary);
        margin: 2rem 0;
    }
    
    /* Footer styling */
    .footer {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-primary);
    }
    
    .footer a {
        color: var(--accent-blue);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .footer a:hover {
        color: var(--text-primary);
        text-shadow: 0 0 10px var(--accent-blue);
    }
    
    /* Loading spinner */
    .stSpinner {
        border-color: var(--accent-blue) !important;
    }
    
    /* Alert styling */
    .stAlert {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(20px) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
        
        .detail-grid {
            grid-template-columns: 1fr;
        }
        
        .block-container {
            padding: 1rem 0.5rem;
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-gradient);
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('crop_recommendation_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'crop_recommendation_model.pkl' and 'scaler.pkl' are in the app directory.")
        return None, None

model, scaler = load_model()

# Enhanced crop dictionary and information
crop_dict = {
    1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut',
    6: 'Papaya', 7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon',
    11: 'Grapes', 12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil',
    16: 'Blackgram', 17: 'Mungbean', 18: 'Mothbeans', 19: 'Pigeonpeas',
    20: 'Kidneybeans', 21: 'Chickpea', 22: 'Coffee'
}

crop_info = {
    'Rice': {'season': 'Kharif (Jun-Oct)', 'water_req': 'High (1200-1800mm)', 'temp_range': '20-35°C', 'ph_range': '5.5-7.0', 'icon': '🌾'},
    'Maize': {'season': 'Kharif/Rabi', 'water_req': 'Medium (600-1000mm)', 'temp_range': '20-30°C', 'ph_range': '6.0-7.5', 'icon': '🌽'},
    'Cotton': {'season': 'Kharif (Apr-Oct)', 'water_req': 'Medium (700-1200mm)', 'temp_range': '20-30°C', 'ph_range': '5.8-8.0', 'icon': '🌸'},
    'Apple': {'season': 'Spring (Mar-May)', 'water_req': 'Medium (800-1200mm)', 'temp_range': '21-24°C', 'ph_range': '6.0-7.0', 'icon': '🍎'},
    'Mango': {'season': 'Summer (Mar-Jun)', 'water_req': 'Medium (750-1200mm)', 'temp_range': '24-27°C', 'ph_range': '5.5-7.5', 'icon': '🥭'},
    'Coffee': {'season': 'Year-round', 'water_req': 'High (1500-2000mm)', 'temp_range': '15-28°C', 'ph_range': '6.0-6.5', 'icon': '☕'},
    'Banana': {'season': 'Year-round', 'water_req': 'High (1200-2000mm)', 'temp_range': '26-30°C', 'ph_range': '6.0-7.5', 'icon': '🍌'},
    'Grapes': {'season': 'Winter (Nov-Feb)', 'water_req': 'Medium (600-800mm)', 'temp_range': '15-25°C', 'ph_range': '6.0-7.0', 'icon': '🍇'},
    'Orange': {'season': 'Winter (Oct-Feb)', 'water_req': 'Medium (800-1200mm)', 'temp_range': '13-26°C', 'ph_range': '6.0-7.5', 'icon': '🍊'},
    'Coconut': {'season': 'Year-round', 'water_req': 'High (1200-2000mm)', 'temp_range': '27-30°C', 'ph_range': '5.2-8.0', 'icon': '🥥'},
}

def create_parameter_gauge(value, min_val, max_val, title, optimal_range=None):
    """Create a beautiful gauge chart for parameters"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'color': '#ffffff'}},
        delta = {'reference': (min_val + max_val) / 2},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickcolor': '#ffffff'},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [min_val, max_val * 0.3], 'color': "rgba(255, 182, 193, 0.3)"},
                {'range': [max_val * 0.3, max_val * 0.7], 'color': "rgba(255, 255, 0, 0.3)"},
                {'range': [max_val * 0.7, max_val], 'color': "rgba(144, 238, 144, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'family': 'Inter'},
        height=200
    )
    
    return fig

def main():
    # Enhanced Header
    st.markdown(r"""
    <div class="main-header">
        <h1>🌾 Smart Crop AI</h1>
        <p>Intelligent Precision Agriculture • AI-Powered Crop Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Introduction
    st.markdown(r"""
<div class="premium-card">
<h2 class="section-header">🚀 Next-Generation Farming Intelligence</h2>
<p style="color: #b8c2cc; font-size: 1.2rem; text-align: center; margin-bottom: 2rem;">
Harness the power of machine learning and data science to optimize your agricultural decisions.
Our AI analyzes soil composition, environmental factors, and climatic conditions to recommend the perfect crop for maximum yield.
</p>

<div class="feature-grid">
<div class="feature-item">
<span class="feature-icon">🧪</span>
<h4 style="color: #81c784; margin-bottom: 0.5rem;">Soil Analysis</h4>
<p style="color: #b8c2cc; margin: 0;">Advanced NPK & pH analysis</p>
</div>
<div class="feature-item">
<span class="feature-icon">🌡️</span>
<h4 style="color: #64b5f6; margin-bottom: 0.5rem;">Climate Intelligence</h4>
<p style="color: #b8c2cc; margin: 0;">Temperature & humidity optimization</p>
</div>
<div class="feature-item">
<span class="feature-icon">🌧️</span>
<h4 style="color: #ba68c8; margin-bottom: 0.5rem;">Rainfall Prediction</h4>
<p style="color: #b8c2cc; margin: 0;">Precipitation pattern analysis</p>
</div>
<div class="feature-item">
<span class="feature-icon">🎯</span>
<h4 style="color: #ffb74d; margin-bottom: 0.5rem;">AI Recommendations</h4>
<p style="color: #b8c2cc; margin: 0;">Machine learning predictions</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    if model is None or scaler is None:
        st.error("Unable to load the model. Please check if the model files exist.")
        return

    # Enhanced Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Crop Prediction", "📊 Analytics Dashboard", "📖 Crop Encyclopedia"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(r"""
            <div class="param-section">
                <h2 class="section-header">🌱 Soil Parameters</h2>
                <p style="color: #b8c2cc; text-align: center; margin-bottom: 2rem;">
                    Configure your soil nutritional composition
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            N = st.slider(
                "🧪 Nitrogen (N) Content", 
                min_value=0, max_value=140, value=50,
                help="Nitrogen content in soil (mg/kg) - Essential for leaf growth and chlorophyll production"
            )
            
            P = st.slider(
                "⚡ Phosphorus (P) Content", 
                min_value=0, max_value=145, value=50,
                help="Phosphorus content in soil (mg/kg) - Crucial for root development and flowering"
            )
            
            K = st.slider(
                "💪 Potassium (K) Content", 
                min_value=0, max_value=205, value=50,
                help="Potassium content in soil (mg/kg) - Important for plant immunity and fruit quality"
            )
            
            ph = st.slider(
                "⚖️ Soil pH Level", 
                min_value=0.0, max_value=14.0, value=7.0, step=0.1,
                help="pH level of the soil - Affects nutrient availability to plants"
            )

        with col2:
            st.markdown(r"""
            <div class="param-section">
                <h2 class="section-header">🌡️ Environmental Conditions</h2>
                <p style="color: #b8c2cc; text-align: center; margin-bottom: 2rem;">
                    Set your local climate parameters
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            temperature = st.slider(
                "🌡️ Average Temperature", 
                min_value=0.0, max_value=50.0, value=25.0,
                help="Average temperature in Celsius - Critical for crop growth and development"
            )
            
            humidity = st.slider(
                "💧 Relative Humidity", 
                min_value=0.0, max_value=100.0, value=50.0,
                help="Relative humidity percentage - Affects disease susceptibility and water needs"
            )
            
            rainfall = st.slider(
                "🌧️ Annual Rainfall", 
                min_value=0.0, max_value=300.0, value=100.0,
                help="Annual rainfall in millimeters - Determines irrigation requirements"
            )

        # Enhanced Prediction Section
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔮 Generate AI Recommendation"):
                try:
                    with st.spinner('🤖 AI is analyzing your agricultural conditions...'):
                        # Simulate some processing time for better UX
                        import time
                        time.sleep(1)
                        
                        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                        transformed_features = scaler.transform(features)
                        prediction = model.predict(transformed_features)
                        predicted_crop = crop_dict.get(prediction[0], "Unknown")

                    # Enhanced Results Display
                    if predicted_crop in crop_info:
                        info = crop_info[predicted_crop]
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h2>🎉 Optimal Crop Recommendation</h2>
                            <h1>{info["icon"]} {predicted_crop}</h1>
                            <p style="font-size: 1.2rem; margin: 0; position: relative; z-index: 1;">
                                Perfect match for your farming conditions!
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Enhanced crop details
                        st.markdown(f"""
                        <div class="premium-card">
                            <h3 class="section-header">📋 Detailed Crop Information</h3>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <h4 style="color: #81c784;">🌱 Growing Season</h4>
                                    <p>{info["season"]}</p>
                                </div>
                                <div class="detail-item">
                                    <h4 style="color: #64b5f6;">💧 Water Requirements</h4>
                                    <p>{info["water_req"]}</p>
                                </div>
                                <div class="detail-item">
                                    <h4 style="color: #ba68c8;">🌡️ Temperature Range</h4>
                                    <p>{info["temp_range"]}</p>
                                </div>
                                <div class="detail-item">
                                    <h4 style="color: #ffb74d;">⚖️ pH Range</h4>
                                    <p>{info["ph_range"]}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ An error occurred during prediction: {str(e)}")

    with tab2:
        st.markdown(r"""
        <div class="premium-card">
            <h2 class="section-header">📊 Real-time Parameter Analysis</h2>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced visualization with gauges
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig1 = create_parameter_gauge(N, 0, 140, "Nitrogen (N)")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig2 = create_parameter_gauge(P, 0, 145, "Phosphorus (P)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig3 = create_parameter_gauge(K, 0, 205, "Potassium (K)")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig4 = create_parameter_gauge(ph, 0, 14, "pH Level")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Enhanced parameter comparison chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create comprehensive analysis chart
        categories = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        values = [N, P, K, temperature, humidity, ph*10, rainfall/3]  # Normalized for better visualization
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='rgba(102, 126, 234, 1)', width=3),
            name='Current Values'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    tickcolor='#ffffff'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    tickcolor='#ffffff'
                )
            ),
            showlegend=True,
            title={
                'text': "🎯 Agricultural Parameters Radar Analysis",
                'x': 0.5,
                'font': {'size': 20, 'color': '#ffffff'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff', 'family': 'Inter'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown(r"""
        <div class="premium-card">
            <h2 class="section-header">📖 Comprehensive Crop Encyclopedia</h2>
            <p style="color: #b8c2cc; text-align: center; font-size: 1.1rem;">
                Explore detailed information about different crops and their requirements
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Create crop cards
        crops_per_row = 3
        crop_list = list(crop_info.keys())
        
        for i in range(0, len(crop_list), crops_per_row):
            cols = st.columns(crops_per_row)
            for j, col in enumerate(cols):
                if i + j < len(crop_list):
                    crop = crop_list[i + j]
                    info = crop_info[crop]
                    
                    with col:
                        st.markdown(f"""
                        <div class="premium-card" style="min-height: 300px;">
                            <div style="text-align: center; margin-bottom: 1rem;">
                                <span style="font-size: 4rem;">{info["icon"]}</span>
                                <h3 style="color: #ffffff; margin: 0.5rem 0;">{crop}</h3>
                            </div>
                            <div style="space-y: 0.5rem;">
                                <p><strong style="color: #81c784;">Season:</strong> <span style="color: #b8c2cc;">{info["season"]}</span></p>
                                <p><strong style="color: #64b5f6;">Water Needs:</strong> <span style="color: #b8c2cc;">{info["water_req"]}</span></p>
                                <p><strong style="color: #ba68c8;">Temperature:</strong> <span style="color: #b8c2cc;">{info["temp_range"]}</span></p>
                                <p><strong style="color: #ffb74d;">pH Range:</strong> <span style="color: #b8c2cc;">{info["ph_range"]}</span></p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # Enhanced Footer
    st.markdown(r"""
    <div class="footer">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🌾 Smart Crop AI</h3>
        <p style="color: #b8c2cc; margin-bottom: 1rem;">
            Empowering farmers with cutting-edge AI technology for sustainable agriculture
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span style="color: #81c784;">🤖 Powered by Machine Learning</span>
            <span style="color: #64b5f6;">📱 Mobile Responsive</span>
            <span style="color: #ba68c8;">🌍 Global Agriculture Support</span>
        </div>
        <p style="margin-top: 1.5rem; color: #b8c2cc;">
            Made with ❤️ for sustainable farming • 
            <a href="https://github.com" target="_blank">View Source Code</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
