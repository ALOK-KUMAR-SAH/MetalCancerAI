import streamlit as st
import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from metalcancerai.modeling.pipeline import MetalCancerModel

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multi-Scenario DNA Analysis", layout="wide")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { border: 1px solid #d1d1d1; padding: 15px; border-radius: 10px; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Multi-Scenario DNA Damage Analysis")
st.write("**IIT Patna Group Project:** Parallel Sensitivity Testing for Single-Subject Diagnostics")
st.markdown("---")

# --- LOAD ASSETS (Model & Preprocessor) ---
@st.cache_resource
def load_artifacts():
    repo_root = Path(__file__).resolve().parent
    models_path = repo_root / "models"
    
    # Load Preprocessor
    preprocessor = joblib.load(models_path / "preprocessor.joblib")
    
    # Detect the number of features expected by the model
    expected_cols = preprocessor.feature_names_in_
    
    # Create a clean dummy to calculate transformed dimensions
    dummy_data = pd.DataFrame(columns=expected_cols)
    dummy_row = {col: (0.0 if preprocessor.transformers_[0][2].count(col) > 0 else 'unknown') 
                 for col in expected_cols}
    dummy_data = pd.concat([dummy_data, pd.DataFrame([dummy_row])], ignore_index=True)
    
    transformed_dummy = preprocessor.transform(dummy_data)
    transformed_dim = transformed_dummy.shape[1]
    
    # Load PyTorch Model
    model = MetalCancerModel(input_dim=transformed_dim)
    model.load_state_dict(torch.load(models_path / "cancer_detector.pth", map_location='cpu'))
    model.eval()
    
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()

    # --- 1. GLOBAL CONSTANTS ---
    st.header("1. Global Physiological Controls")
    c1, c2, c3 = st.columns(3)
    with c1: ph = st.number_input("pH Level", value=7.4, format="%.2f")
    with c2: temp = st.number_input("Body Temp (°C)", value=37.0, format="%.1f")
    with c3: area = st.number_input("Peak Area", value=1.0, format="%.2f")

    st.markdown("---")
    
    # --- 2. MULTI-SCENARIO INPUTS ---
    st.header("2. Scenario-Specific Variable Inputs")
    
    scenarios = {
        "Baseline (Healthy)": {"ip": 0.60, "ep": 50.0, "age": 30.0, "inf": 0, "diab": 0},
        "A: Biological Age": {"ip": 0.30, "ep": 80.0, "age": 70.0, "inf": 1, "diab": 0},
        "B: Industrial Exp.": {"ip": 0.95, "ep": 20.0, "age": 35.0, "inf": 0, "diab": 0},
        "C: Metabolic Stress": {"ip": 0.45, "ep": 65.0, "age": 35.0, "inf": 1, "diab": 1}
    }

    cols = st.columns(4)
    final_inputs = {}

    for i, (name, defaults) in enumerate(scenarios.items()):
        with cols[i]:
            st.subheader(name)
            ip_val = st.slider(f"ip ({name})", 0.0, 1.0, defaults["ip"], key=f"ip_{i}")
            ep_val = st.slider(f"ep ({name})", 0.0, 100.0, defaults["ep"], key=f"ep_{i}")
            final_inputs[name] = {
                'ip': ip_val, 'ep': ep_val, 'age': defaults["age"], 
                'inf': defaults["inf"], 'diab': defaults["diab"]
            }

    # --- 3. EXECUTION ---
    if st.button("🚀 Run Comprehensive Parallel Analysis", use_container_width=True):
        all_results = []
        
        for name, vals in final_inputs.items():
            # Build the exact dictionary expected by your preprocessor
            input_dict = {
                'ip': [float(vals['ip'])], 
                'ep': [float(vals['ep'])], 
                'peak_width': [0.2], 
                'area': [float(area)], 
                'baseline': [0.1],
                'ph': [float(ph)], 
                'temperature': [float(temp)], 
                'sample_type': ['saliva'], 
                'age': [float(vals['age'])], 
                'sex': ['m'], 
                'diabetes': [int(vals['diab'])],
                'ckd': [0], 
                'cvd': [0], 
                'cancer': [0], 
                'inflammation': [int(vals['inf'])]
            }
            
            # Convert to DataFrame and fix column order to match preprocessor
            input_df = pd.DataFrame(input_dict)
            input_df = input_df[preprocessor.feature_names_in_]
            
            # String Sanitization
            for col in input_df.select_dtypes(include=['object']).columns:
                input_df[col] = input_df[col].astype(str).str.lower().str.strip()

            # Transform and check for NaNs manually to prevent ufunc error
            X_scaled = preprocessor.transform(input_df)
            X_scaled = np.nan_to_num(X_scaled) # Final safety check
            
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                risk_tensor, time_tensor = model(X_tensor)
            
            risk_pct = max(0.0, min(100.0, float(risk_tensor.item()) * 100))
            
            all_results.append({
                "Scenario": name, 
                "Risk (%)": round(risk_pct, 2), 
                "Onset (Months)": round(float(time_tensor.item()), 1)
            })

        # --- 4. VISUALIZATION ---
        res_df = pd.DataFrame(all_results)
        st.markdown("---")
        st.header("3. Comparative Diagnostic Insights")
        
        res_left, res_right = st.columns([1, 1.5])
        
        with res_left:
            st.write("#### Results Summary")
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            max_idx = res_df['Risk (%)'].idxmax()
            st.error(f"**Highest Risk Detected:** {res_df.iloc[max_idx]['Scenario']}")

        with res_right:
            st.write("#### Risk Variance Visualization")
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#27ae60', '#e74c3c', '#2980b9', '#f39c12']
            ax.bar(res_df["Scenario"], res_df["Risk (%)"], color=colors)
            ax.set_ylabel("DNA Damage Risk (%)")
            ax.set_ylim(0, 100)
            st.pyplot(fig)

        st.success("Analysis Complete.")

except Exception as e:
    st.error(f"System Error: {e}")