import streamlit as st
import torch
import pandas as pd
import joblib
from pathlib import Path
from metalcancerai.modeling.pipeline import MetalCancerModel

# Page Config - Updated name
st.set_page_config(page_title="Heavy Metal DNA Damage AI", layout="wide")

# Main Title - Updated name
st.title("🔬 Heavy Metal Induced DNA Damage Detection System")
st.markdown("---")

# Load Model & Scaler
@st.cache_resource
def load_artifacts():
    repo_root = Path(__file__).resolve().parent
    scaler = joblib.load(repo_root / "models" / "scaler.joblib")
    
    # Auto-detect input dim
    input_dim = scaler.transform(pd.DataFrame([[0]*len(scaler.feature_names_in_)], 
                                 columns=scaler.feature_names_in_)).shape[1]
    
    model = MetalCancerModel(input_dim=input_dim)
    model.load_state_dict(torch.load(repo_root / "models" / "cancer_detector.pth", map_location='cpu'))
    model.eval()
    return model, scaler

try:
    model, scaler = load_artifacts()

    # Sidebar for Inputs
    st.sidebar.header("📍 Patient & Sensor Data")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sensor Readings")
        ip = st.slider("Peak Current (ip)", 0.0, 1.0, 0.5)
        ep = st.slider("Peak Potential (ep)", 0.0, 100.0, 50.0)
        ph = st.slider("pH Level", 0.0, 14.0, 7.0)
        temp = st.slider("Temperature (°C)", 30.0, 45.0, 37.0)
        area = st.number_input("Peak Area", value=1.0)

    with col2:
        st.subheader("Patient Demographics")
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        sex = st.selectbox("Sex", ["M", "F"])
        sample = st.selectbox("Sample Type", ["urine", "saliva"])
        diabetes = st.checkbox("Diabetes")
        inflammation = st.checkbox("Inflammation")

    # Prediction Logic
    if st.button("🚀 Run DNA Damage Analysis", use_container_width=True):
        # Prepare Data
        input_dict = {
            'ip': [ip], 'ep': [ep], 'peak_width': [0.2], 'area': [area], 'baseline': [0.1],
            'ph': [ph], 'temperature': [temp], 'sample_type': [sample],
            'age': [age], 'sex': [sex], 'diabetes': [1 if diabetes else 0],
            'ckd': [0], 'cvd': [0], 'cancer': [0], 'inflammation': [1 if inflammation else 0]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Transform & Predict
        X_scaled = scaler.transform(input_df)
        with torch.no_grad():
            risk_raw, time_raw = model(torch.tensor(X_scaled, dtype=torch.float32))
        
        risk = torch.sigmoid(risk_raw/100).item() * 100
        time = max(1, round(abs(time_raw.item() % 60)))

        # Display Results
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("DNA Damage Risk Score", f"{risk:.2f}%")
            if risk > 70: 
                st.error("CRITICAL DAMAGE DETECTED")
            elif risk > 30: 
                st.warning("MODERATE DAMAGE DETECTED")
            else: 
                st.success("LOW/MINIMAL DAMAGE")
            
        with res_col2:
            st.metric("Est. Time to Clinical Symptoms", f"{time} Months")
            st.info("Continuous monitoring recommended based on DNA toxicity levels.")

except Exception as e:
    st.error(f"Please run training first! Error: {e}")