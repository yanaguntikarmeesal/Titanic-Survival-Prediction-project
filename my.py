# ==============================================================
# 🚢 TITANIC SURVIVAL PREDICTION
# Logistic Regression Machine Learning Project
# Developed by Yanaguntikar Meesal
# ==============================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ==============================================================
# 1. PAGE CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================================================
# 2. CUSTOM CSS STYLING
# ==============================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Playfair+Display:wght@700;900&display=swap');

    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #1a2f4f 50%, #0d2137 100%);
        font-family: 'Poppins', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 850px;
    }

    /* Hero Header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.9), rgba(15, 32, 58, 0.95));
        border-radius: 20px;
        border: 1px solid rgba(100, 181, 246, 0.3);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                    0 0 40px rgba(100, 181, 246, 0.1) inset;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(100, 181, 246, 0.08) 0%, transparent 70%);
        animation: shimmer 8s infinite linear;
    }

    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #64b5f6, #ffffff, #90caf9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: 1px;
    }

    .hero-subtitle {
        color: #90caf9;
        font-size: 1.05rem;
        font-weight: 300;
        margin-top: 0.8rem;
        position: relative;
        z-index: 1;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #64b5f6, transparent);
        margin: 1.2rem auto 0;
        border-radius: 2px;
        position: relative;
        z-index: 1;
    }

    /* Section Headers */
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #e3f2fd;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid rgba(100, 181, 246, 0.3);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* Input Card */
    .input-card {
        background: rgba(20, 35, 60, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.8rem;
        border: 1px solid rgba(100, 181, 246, 0.15);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .input-card:hover {
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4),
                    0 0 20px rgba(100, 181, 246, 0.1);
    }

    /* Streamlit Widgets Styling */
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stRadio label {
        color: #e3f2fd !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* Selectbox, Number input */
    div[data-baseweb="select"] > div,
    .stNumberInput input {
        background-color: rgba(30, 50, 80, 0.8) !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        border-radius: 10px !important;
        color: #e3f2fd !important;
    }

    div[data-baseweb="select"] > div:hover,
    .stNumberInput input:hover {
        border-color: rgba(100, 181, 246, 0.6) !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: rgba(100, 181, 246, 0.2) !important;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #64b5f6, #42a5f5) !important;
    }

    /* Predict Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 50%, #64b5f6 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 1.5px;
        padding: 1rem 2rem;
        border: none;
        border-radius: 50px;
        box-shadow: 0 8px 25px rgba(66, 165, 245, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(66, 165, 245, 0.6);
        background: linear-gradient(135deg, #1565c0 0%, #1e88e5 50%, #42a5f5 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Result Cards */
    .result-card {
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }

    .result-survived {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.3), rgba(27, 94, 32, 0.4));
        border: 2px solid rgba(76, 175, 80, 0.5);
        box-shadow: 0 0 40px rgba(76, 175, 80, 0.3);
    }

    .result-not-survived {
        background: linear-gradient(135deg, rgba(198, 40, 40, 0.3), rgba(183, 28, 28, 0.4));
        border: 2px solid rgba(244, 67, 54, 0.5);
        box-shadow: 0 0 40px rgba(244, 67, 54, 0.3);
    }

    .result-icon {
        font-size: 4.5rem;
        margin-bottom: 1rem;
        display: block;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-title {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .result-survived .result-title {
        color: #81c784;
    }

    .result-not-survived .result-title {
        color: #ef5350;
    }

    .result-message {
        color: #cfd8dc;
        font-size: 1rem;
        font-weight: 300;
        line-height: 1.7;
    }

    /* Probability badge */
    .prob-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        margin-top: 1rem;
        font-weight: 500;
        font-size: 0.95rem;
        color: #e3f2fd;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem 1rem;
        border-top: 1px solid rgba(100, 181, 246, 0.15);
        color: #78909c;
        font-size: 0.85rem;
        font-weight: 300;
    }

    .footer .dev-name {
        color: #64b5f6;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* Info box */
    .info-box {
        background: rgba(21, 101, 192, 0.1);
        border-left: 4px solid #42a5f5;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #bbdefb;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        font-weight: 300;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Radio button styling */
    .stRadio > div {
        background: transparent;
        gap: 1rem;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero-subtitle { font-size: 0.85rem; letter-spacing: 1px; }
        .result-title { font-size: 1.5rem; }
        .result-icon { font-size: 3.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# 3. HERO HEADER
# ==============================================================

st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🚢 Titanic Survival</h1>
    <p class="hero-subtitle">Logistic Regression Prediction Engine</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ==============================================================
# 4. LOAD MACHINE LEARNING MODEL
# ==============================================================

@st.cache_resource
def load_model():
    try:
        with open('titanic.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'titanic_model.pkl' not found. Please ensure it's in the same directory.")
        return None

model = load_model()

# ==============================================================
# 5. INPUT SECTION
# ==============================================================

st.markdown('<div class="section-title">📋 Passenger Information</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">💡 Enter the passenger details below to predict their survival probability aboard the RMS Titanic.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "🎫 Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"{'First' if x==1 else 'Second' if x==2 else 'Third'} Class",
        help="Ticket class of the passenger"
    )
    
    sex = st.selectbox(
        "👤 Gender",
        options=["male", "female"],
        format_func=lambda x: "♂ Male" if x == "male" else "♀ Female"
    )
    
    age = st.number_input(
        "🎂 Age",
        min_value=0,
        max_value=100,
        value=0,
        help="Age of the passenger in years"
    )

with col2:
    sibsp = st.number_input(
        "👨‍👩‍👧 Siblings / Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of siblings or spouses traveling with the passenger"
    )
    
    parch = st.number_input(
        "👶 Parents / Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of parents or children traveling with the passenger"
    )
    
    fare = st.number_input(
        "💰 Fare Paid (£)",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=0.5,
        help="Passenger fare in British Pounds"
    )

embarked = st.selectbox(
    "⚓ Port of Embarkation",
    options=["C", "Q", "S"],
    format_func=lambda x: {"C": "🇫🇷 Cherbourg", "Q": "🇮🇪 Queenstown", "S": "🇬🇧 Southampton"}[x]
)

# ==============================================================
# 6. PREDICTION
# ==============================================================

if st.button("🔮 Predict Survival", use_container_width=True):
    if model is not None:
        # Encode inputs
        sex_encoded = 1 if sex == "male" else 0
        embarked_encoded = {"S": 0, "C": 1, "Q": 2}[embarked]
        
        # Prepare features (adjust order to match your model's training)
        features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
        
        try:
            prediction = model.predict(features)[0]
            
            # Get probability if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                confidence = proba[int(prediction)] * 100
                prob_text = f"Confidence: {confidence:.1f}%"
            else:
                prob_text = "Prediction complete"
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card result-survived">
                    <span class="result-icon">🎉</span>
                    <div class="result-title">Would Survive</div>
                    <div class="result-message">
                        Based on the provided information, this passenger<br>
                        is predicted to have <strong>survived</strong> the Titanic disaster.
                    </div>
                    <div class="prob-badge">{prob_text}</div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-card result-not-survived">
                    <span class="result-icon">🌊</span>
                    <div class="result-title">Would Not Survive</div>
                    <div class="result-message">
                        Based on the provided information, this passenger<br>
                        is predicted to have <strong>not survived</strong> the Titanic disaster.
                    </div>
                    <div class="prob-badge">{prob_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Make sure the model expects features in this order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")

# ==============================================================
# 7. FOOTER
# ==============================================================

st.markdown("""
<div class="footer">
    <p>⚓ Developed with <span style="color:#ef5350;">♥</span> by <span class="dev-name">Yanaguntikar Meesal</span></p>
    <p style="margin-top:0.5rem; font-size:0.75rem; opacity:0.7;">Powered by Logistic Regression · Streamlit</p>
</div>
""", unsafe_allow_html=True)