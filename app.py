import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Page Configuration & Device-Friendly Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioPulse - Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="auto"  # Automatically collapses on mobile, expands on desktop
)

# Responsive, device-friendly CSS with mobile-first and fluid media queries
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Device-friendly page container padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1300px;
    }

    /* Fluid Responsive Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: clamp(16px, 3vw, 28px);
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 20px;
        border-left: 6px solid #ef4444;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: clamp(1.35rem, 3.2vw, 2.2rem) !important;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #ffffff !important;
        line-height: 1.25;
    }
    
    .main-header p {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: clamp(0.85rem, 1.6vw, 1.05rem);
        line-height: 1.5;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: clamp(0.72rem, 1.2vw, 0.82rem);
        font-weight: 600;
        white-space: nowrap;
    }
    
    .badge-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.35); }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }

    /* Section Card Wrappers */
    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: clamp(14px, 2vw, 20px);
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        height: 100%;
    }

    .input-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
    }

    /* Result Card Styles */
    .result-card-danger {
        background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: clamp(16px, 3vw, 24px);
        color: #991b1b;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.18);
        margin-bottom: 16px;
    }

    .result-card-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: clamp(16px, 3vw, 24px);
        color: #166534;
        box-shadow: 0 10px 25px -5px rgba(34, 197, 94, 0.18);
        margin-bottom: 16px;
    }

    .metric-value {
        font-size: clamp(1.3rem, 2.5vw, 2.1rem);
        font-weight: 800;
        line-height: 1.2;
        margin-top: 4px;
    }

    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        opacity: 0.85;
    }

    .stats-row {
        margin-top: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: clamp(12px, 3vw, 24px);
    }

    .stat-item {
        flex: 1 1 120px;
        background: rgba(255, 255, 255, 0.6);
        padding: 10px 14px;
        border-radius: 10px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    .stat-number {
        font-size: clamp(1.3rem, 2.2vw, 1.7rem);
        font-weight: 800;
    }

    /* Device-Friendly Touch Targets & Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white !important;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: clamp(12px, 2vw, 16px) 24px;
        font-size: clamp(1rem, 1.8vw, 1.15rem);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35);
        width: 100%;
        min-height: 48px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.45);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    /* Quick Preset Chips on Top */
    .preset-chip-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 6px;
    }

    /* -------------------------------------------------- */
    /* RESPONSIVE MEDIA QUERIES FOR MOBILE & TABLETS     */
    /* -------------------------------------------------- */
    @media (max-width: 768px) {
        /* Mobile padding adjustments */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        /* Stack horizontal layout blocks cleanly on mobile */
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.75rem !important;
        }
        
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
        }

        /* Better touch area for mobile selectbox & inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            min-height: 44px !important;
        }

        .stats-row {
            flex-direction: column !important;
            gap: 8px !important;
        }

        .stat-item {
            width: 100% !important;
        }
    }

    @media (min-width: 769px) and (max-width: 1024px) {
        /* Tablet adjustments */
        .block-container {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Artifacts (Model, Scaler, Columns)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "KNN_heart.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_FILE = os.path.join(BASE_DIR, "columns.pkl")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        columns = joblib.load(COLUMNS_FILE)
        return model, scaler, columns, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, feature_columns, load_error = load_assets()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>❤️ CardioPulse — Heart Disease Risk Analysis</h1>
    <p>Clinical decision support tool powered by a trained K-Nearest Neighbors (KNN) Machine Learning Model</p>
    <div>
        <span class="badge-pill badge-blue">Model: KNN Classifier</span>
        <span class="badge-pill badge-green">StandardScaler Pipeline</span>
        <span class="badge-pill badge-purple">15 Feature Dimensions</span>
    </div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"❌ Error loading model artifacts: {load_error}")
    st.stop()

# ---------------------------------------------------------
# Sidebar & Main Quick Profile Controls (Mobile & Desktop Friendly)
# ---------------------------------------------------------
preset_options = [
    "Custom Input",
    "Sample 1: Healthy Profile (Low Risk)",
    "Sample 2: Cardiac Patient (High Risk)"
]

# Ensure session state for preset synchronization
if "preset_selection" not in st.session_state:
    st.session_state.preset_selection = "Custom Input"

def sync_from_sidebar():
    st.session_state.preset_selection = st.session_state.sidebar_preset

def sync_from_main():
    st.session_state.preset_selection = st.session_state.main_preset

with st.sidebar:
    st.header("⚙️ Patient Profiles & Controls")
    st.markdown("Quickly test with sample patient records:")
    st.radio(
        "Load Clinical Preset:",
        preset_options,
        index=preset_options.index(st.session_state.preset_selection),
        key="sidebar_preset",
        on_change=sync_from_sidebar
    )

    st.markdown("---")
    st.subheader("📋 Pipeline Summary")
    st.markdown("""
    - **Classifier**: `KNeighborsClassifier` (k=5)
    - **Scaling**: `StandardScaler`
    - **One-Hot Encoding**: Handled dynamically to match `columns.pkl`
    - **Target**: `0` = Healthy / Normal, `1` = Heart Disease
    """)

    st.markdown("---")
    st.caption("🔒 Medical Disclaimer: For educational and exploratory clinical analysis purposes only.")

# On mobile, show a compact preset bar directly in the main view
preset = st.session_state.preset_selection

with st.expander("⚡ Quick Load Patient Preset (Tap to Select)", expanded=(preset != "Custom Input")):
    st.selectbox(
        "Choose a clinical profile to auto-populate the form:",
        preset_options,
        index=preset_options.index(preset),
        key="main_preset",
        on_change=sync_from_main,
        help="Quickly pre-fill diagnostic parameters for instant testing"
    )

# Preset default definitions
if preset == "Sample 1: Healthy Profile (Low Risk)":
    default_age = 40
    default_sex = "Male"
    default_cp = "ATA (Atypical Angina)"
    default_trestbps = 140
    default_chol = 289
    default_fbs = "No (<= 120 mg/dl)"
    default_restecg = "Normal"
    default_thalach = 172
    default_exang = "No"
    default_oldpeak = 0.0
    default_slope = "Up (Upsloping)"
elif preset == "Sample 2: Cardiac Patient (High Risk)":
    default_age = 58
    default_sex = "Male"
    default_cp = "ASY (Asymptomatic)"
    default_trestbps = 150
    default_chol = 270
    default_fbs = "Yes (> 120 mg/dl)"
    default_restecg = "ST (ST-T abnormality)"
    default_thalach = 112
    default_exang = "Yes"
    default_oldpeak = 2.5
    default_slope = "Flat"
else:
    default_age = 54
    default_sex = "Male"
    default_cp = "NAP (Non-Anginal Pain)"
    default_trestbps = 130
    default_chol = 220
    default_fbs = "No (<= 120 mg/dl)"
    default_restecg = "Normal"
    default_thalach = 145
    default_exang = "No"
    default_oldpeak = 0.5
    default_slope = "Up (Upsloping)"

# ---------------------------------------------------------
# Patient Input Form (Device-Adaptive Grid)
# ---------------------------------------------------------
st.markdown("### 🩺 Diagnostic Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="input-card-title">
        <span>👤 Demographics & Vitals</span>
    </div>
    """, unsafe_allow_html=True)
    
    age = st.slider(
        "Age (years)", 
        min_value=20, 
        max_value=90, 
        value=default_age, 
        step=1,
        help="Patient's chronological age in years"
    )
    
    sex = st.selectbox(
        "Biological Sex", 
        options=["Male", "Female"],
        index=0 if default_sex == "Male" else 1,
        help="Biological sex of the patient"
    )
    
    trestbps = st.number_input(
        "Resting BP (mm Hg)", 
        min_value=70, 
        max_value=220,
        value=int(default_trestbps), 
        step=1,
        help="Blood pressure measured in mm Hg at clinical intake"
    )

with col2:
    st.markdown("""
    <div class="input-card-title">
        <span>🧪 Serum & Metabolic</span>
    </div>
    """, unsafe_allow_html=True)
    
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)", 
        min_value=0, 
        max_value=650,
        value=int(default_chol), 
        step=5,
        help="Serum cholesterol in mg/dl (0 if unmeasured)"
    )
    
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
        index=0 if "No" in default_fbs else 1,
        help="Indicates elevated fasting glycemia"
    )
    
    thalach = st.number_input(
        "Max Heart Rate Achieved", 
        min_value=60, 
        max_value=220,
        value=int(default_thalach), 
        step=1,
        help="Peak heart rate recorded during exercise/stress test"
    )

with col3:
    st.markdown("""
    <div class="input-card-title">
        <span>⚡ ECG & Stress Testing</span>
    </div>
    """, unsafe_allow_html=True)
    
    cp = st.selectbox(
        "Chest Pain Type",
        options=["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)", "TA (Typical Angina)"],
        index=["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)", "TA (Typical Angina)"].index(default_cp),
        help="Subjective chest pain symptomatology"
    )

    restecg = st.selectbox(
        "Resting Electrocardiogram",
        options=["Normal", "ST (ST-T abnormality)", "LVH (Left Ventricular Hypertrophy)"],
        index=["Normal", "ST (ST-T abnormality)", "LVH (Left Ventricular Hypertrophy)"].index(default_restecg),
        help="Baseline electrocardiogram status"
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=["No", "Yes"],
        index=0 if default_exang == "No" else 1,
        help="Exertional angina provoked by exercise"
    )

    oldpeak = st.slider(
        "ST Depression (Oldpeak)", 
        min_value=-2.5, 
        max_value=6.5,
        value=float(default_oldpeak), 
        step=0.1,
        help="ST depression in mm induced by exercise relative to rest"
    )

    slope = st.selectbox(
        "Peak ST Segment Slope",
        options=["Up (Upsloping)", "Flat", "Down (Downsloping)"],
        index=["Up (Upsloping)", "Flat", "Down (Downsloping)"].index(default_slope),
        help="Slope trend of ST segment during maximum physical exertion"
    )

# ---------------------------------------------------------
# Feature Preprocessing & Transformation
# ---------------------------------------------------------
# Map inputs into one-hot encoded binary columns matching training encoding (drop_first=True)
sex_m = 1 if sex == "Male" else 0
cp_ata = 1 if "ATA" in cp else 0
cp_nap = 1 if "NAP" in cp else 0
cp_ta = 1 if "TA" in cp and "ATA" not in cp else 0
# Notice: If cp == "ASY", cp_ata = cp_nap = cp_ta = 0 (baseline dropped category)

restecg_normal = 1 if restecg == "Normal" else 0
restecg_st = 1 if "ST" in restecg else 0
# Notice: If restecg == "LVH", both restecg_normal and restecg_st = 0 (baseline dropped category)

exang_y = 1 if exang == "Yes" else 0

st_slope_flat = 1 if slope == "Flat" else 0
st_slope_up = 1 if "Up" in slope else 0
# Notice: If slope == "Down", both st_slope_flat and st_slope_up = 0 (baseline dropped category)

fbs_val = 1 if "Yes" in fbs else 0

# Baseline dataset numerical normalization stats from training:
NUMERICAL_STATS = {
    'Age': {'mean': 53.510893, 'std': 9.427478},
    'RestingBP': {'mean': 132.396514, 'std': 18.504067},
    'Cholesterol': {'mean': 198.799564, 'std': 109.324551},
    'MaxHR': {'mean': 136.809368, 'std': 25.446463},
    'Oldpeak': {'mean': 0.887364, 'std': 1.065989}
}

# Construct dataframe row matching exact column names in columns.pkl
input_dict = {
    'Age': (age - NUMERICAL_STATS['Age']['mean']) / NUMERICAL_STATS['Age']['std'],
    'RestingBP': (trestbps - NUMERICAL_STATS['RestingBP']['mean']) / NUMERICAL_STATS['RestingBP']['std'],
    'Cholesterol': (chol - NUMERICAL_STATS['Cholesterol']['mean']) / NUMERICAL_STATS['Cholesterol']['std'],
    'FastingBS': fbs_val,
    'MaxHR': (thalach - NUMERICAL_STATS['MaxHR']['mean']) / NUMERICAL_STATS['MaxHR']['std'],
    'Oldpeak': (oldpeak - NUMERICAL_STATS['Oldpeak']['mean']) / NUMERICAL_STATS['Oldpeak']['std'],
    'Sex_M': sex_m,
    'ChestPainType_ATA': cp_ata,
    'ChestPainType_NAP': cp_nap,
    'ChestPainType_TA': cp_ta,
    'RestingECG_Normal': restecg_normal,
    'RestingECG_ST': restecg_st,
    'ExerciseAngina_Y': exang_y,
    'ST_Slope_Flat': st_slope_flat,
    'ST_Slope_Up': st_slope_up
}

input_df = pd.DataFrame([input_dict])[feature_columns]

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
predict_clicked = st.button("🚀 Analyze Heart Disease Risk", use_container_width=True)

# ---------------------------------------------------------
# Prediction Execution & Results Visualization
# ---------------------------------------------------------
if predict_clicked:
    try:
        # Scale data using the saved scaler.pkl
        scaled_input = scaler.transform(input_df)
        
        # Make prediction with KNN model
        prediction = model.predict(scaled_input)[0]
        
        # Get class probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            prob_healthy = probabilities[0] * 100
            prob_disease = probabilities[1] * 100
        else:
            prob_disease = 100.0 if prediction == 1 else 0.0
            prob_healthy = 100.0 - prob_disease
            
        st.markdown("---")
        st.markdown("### 📊 Diagnostic Assessment Results")
        
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-danger">
                    <div class="metric-label">Diagnostic Status</div>
                    <div class="metric-value">⚠️ High Risk of Heart Disease</div>
                    <p style="margin-top: 8px; line-height: 1.5; font-size: 0.95rem;">
                        The trained model indicates clinical patterns strongly associated with cardiovascular impairment.
                    </p>
                    <div class="stats-row">
                        <div class="stat-item">
                            <div class="metric-label">Disease Probability</div>
                            <div class="stat-number" style="color: #b91c1c;">{prob_disease:.1f}%</div>
                        </div>
                        <div class="stat-item">
                            <div class="metric-label">Confidence Score</div>
                            <div class="stat-number" style="color: #1e293b;">{max(prob_healthy, prob_disease):.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-success">
                    <div class="metric-label">Diagnostic Status</div>
                    <div class="metric-value">✅ Low Risk / Normal Profile</div>
                    <p style="margin-top: 8px; line-height: 1.5; font-size: 0.95rem;">
                        The submitted cardiovascular biomarkers align with lower clinical risk according to the KNN model.
                    </p>
                    <div class="stats-row">
                        <div class="stat-item">
                            <div class="metric-label">Healthy Probability</div>
                            <div class="stat-number" style="color: #15803d;">{prob_healthy:.1f}%</div>
                        </div>
                        <div class="stat-item">
                            <div class="metric-label">Confidence Score</div>
                            <div class="stat-number" style="color: #1e293b;">{max(prob_healthy, prob_disease):.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("##### 📈 Probability Breakdown")
            chart_data = pd.DataFrame({
                "Category": ["Low Risk / Normal", "Heart Disease Risk"],
                "Probability (%)": [prob_healthy, prob_disease]
            }).set_index("Category")
            
            st.bar_chart(chart_data, color=["#ef4444" if prediction == 1 else "#22c55e"])
            
            # Key observation highlights
            st.markdown("##### 🔍 Key Risk Factors Flagged:")
            flags = []
            if trestbps > 135:
                flags.append(f"• **Elevated Resting BP**: {trestbps} mm Hg (Hypertension threshold)")
            if chol > 240:
                flags.append(f"• **High Serum Cholesterol**: {chol} mg/dl (Target < 200 mg/dl)")
            if fbs_val == 1:
                flags.append("• **Elevated Fasting Blood Sugar**: > 120 mg/dl")
            if exang_y == 1:
                flags.append("• **Exercise-Induced Angina**: Positive exertion pain")
            if oldpeak > 1.5:
                flags.append(f"• **Significant ST Depression**: {oldpeak} mm")
            if "Flat" in slope or "Down" in slope:
                flags.append(f"• **Abnormal ST Slope**: {slope}")
            if "ASY" in cp:
                flags.append("• **Asymptomatic Chest Pain**: Statistically strong correlate for silent ischemia")
                
            if flags:
                for f in flags:
                    st.write(f)
            else:
                st.write("• No critical high-risk flags identified in submitted parameters.")

        # Detailed breakdown expander
        with st.expander("🛠️ View Scaled Features & Model Vector"):
            st.write("Exact 15-dimensional vector fed into `scaler.pkl` and `KNN_heart.pkl`:")
            debug_df = pd.DataFrame({
                "Feature Column": feature_columns,
                "Raw / Preprocessed Value": input_df.values[0],
                "StandardScaled Value": scaled_input[0]
            })
            st.dataframe(debug_df, use_container_width=True)

    except Exception as err:
        st.error(f"Prediction failed: {err}")
        st.exception(err)
