import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Health AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'heart_disease_model.joblib' and 'scaler.joblib' are in the same directory.")
        return None, None

# ---------------- HOME PAGE ----------------
def show_home_page():
    st.markdown("""
    <div class="main-header">
        <h1>Heart Health Prediction Platform</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">AI-powered risk assessment for heart disease</p>
    </div>
    """, unsafe_allow_html=True)

    # Features Section using native Streamlit components
    st.markdown("## Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**AI-Based Prediction**\n\nUtilizes machine learning trained on the UCI Heart Disease dataset with ~85% accuracy using Logistic Regression.")
    
    with col2:
        st.success("**Confidence Scoring**\n\nGet probability-based confidence scores with every prediction result for better understanding.")

    # How It Works Section
    st.markdown("---")
    st.markdown("## How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Data Collection")
        st.write("Model trained on UCI Heart Disease dataset using validated medical indicators.")
    
    with col2:
        st.markdown("### 2️⃣ Standardization")
        st.write("Input features standardized using StandardScaler for optimal performance.")
    
    with col3:
        st.markdown("### 3️⃣ Risk Prediction")
        st.write("Model outputs risk prediction with probability confidence score.")

    # About Section
    st.markdown("---")
    st.markdown("## About This Platform")
    
    st.info("""
    This interactive web application estimates your risk of heart disease based on standard medical health indicators. 
    The tool uses a scientifically validated machine learning model to provide personalized risk assessments.
    
    **Features:**
    - AI-based Heart Disease Risk Prediction
    - Confidence Score for predictions
    - Detailed explanation of all input medical terms
    - User-friendly interface with guided inputs
    """)

    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Important Disclaimer**
    
    It is **not a medical diagnosis** 
    and should not replace professional medical advice. Always consult a **healthcare professional** 
    for clinical advice and proper medical evaluation.
    """)

    # CTA Button - Centered
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Get Started with Prediction", key="home_cta", use_container_width=True):
            st.session_state.page = "prediction"
            st.rerun()

# ---------------- PREDICTION PAGE ----------------
def show_prediction_page(model, scaler):
    if model is None or scaler is None:
        st.error("Unable to load model. Please check if model files exist.")
        return

    st.markdown("""
    <div class="main-header">
        <h1>Heart Disease Prediction Tool</h1>
        <p style="font-size: 1.1rem; margin-top: 1rem;">Enter your health details for risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Info Sections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("User Guidelines", expanded=False):
            st.markdown("""
            **Welcome to the Prediction Tool!**
            
            - Enter all health parameters accurately
            - All fields are required for prediction
            - Click " Predict Risk" to view results
            - This is for educational purposes only
            - Always consult a healthcare professional
            """)
    
    with col2:
        with st.expander("Model Information", expanded=False):
            st.markdown("""
            **Technical Details:**
            
            - **Algorithm:** Logistic Regression
            - **Dataset:** UCI Heart Disease Dataset
            - **Accuracy:** ~85% on validation data
            - **Framework:** Scikit-learn
            - **Preprocessing:** StandardScaler
            """)
    
    with col3:
        with st.expander("Terms Explained", expanded=False):
            st.markdown("""
            **Key Medical Terms:**
            
            - **cp:** Chest pain type (0-3)
            - **trestbps:** Resting blood pressure
            - **chol:** Serum cholesterol
            - **thalach:** Maximum heart rate
            - **oldpeak:** ST depression value
            
            *See full descriptions in form labels*
            """)

    st.markdown("---")
    
    # Input Form
    st.markdown("## Enter Your Health Parameters")
    
    with st.form("prediction_form"):
        # Row 1: Basic Info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, 
                                help="Your age in years")
        
        with col2:
            sex = st.selectbox("Sex", options=[0, 1], 
                             format_func=lambda x: "Female" if x == 0 else "Male",
                             help="Biological sex: 0 = Female, 1 = Male")
        
        with col3:
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                  "Non-anginal Pain", "Asymptomatic"][x],
                            help="Type of chest pain experienced")

        # Row 2: Vital Signs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                      min_value=80, max_value=200, value=120,
                                      help="Blood pressure at rest")
        
        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", 
                                  min_value=100, max_value=600, value=200,
                                  help="Cholesterol level in blood")
        
        with col3:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes",
                             help="Is fasting blood sugar greater than 120 mg/dl?")

        # Row 3: ECG and Heart Rate
        col1, col2, col3 = st.columns(3)
        
        with col1:
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                 format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
                                                       "Left Ventricular Hypertrophy"][x],
                                 help="Results of resting electrocardiogram")
        
        with col2:
            thalach = st.number_input("Maximum Heart Rate Achieved", 
                                     min_value=60, max_value=220, value=150,
                                     help="Highest heart rate during exercise")
        
        with col3:
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               help="Chest pain during exercise")

        # Row 4: Advanced Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            oldpeak = st.number_input("ST Depression (oldpeak)", 
                                     min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                     help="ST depression induced by exercise")
        
        with col2:
            slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                               help="Slope of peak exercise ST segment")
        
        with col3:
            ca = st.selectbox("Major Vessels (0-4)", options=[0, 1, 2, 3, 4],
                            help="Number of major vessels colored by fluoroscopy")
        
        with col4:
            thal = st.selectbox("Thalassemia", options=[1, 2, 3],
                              format_func=lambda x: ["Normal", "Fixed Defect", 
                                                    "Reversible Defect"][x-1],
                              help="Thalassemia blood disorder type")

        # Submit Button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    # Process Prediction
    if submitted:
        user_input = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }

        # Prepare input for model
        input_array = np.array(list(user_input.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Make prediction
        with st.spinner('Analyzing your health data...'):
            prediction = model.predict(input_scaled)[0]
            confidence = model.predict_proba(input_scaled)[0][int(prediction)] * 100
            result_text = "At Risk" if prediction == 1 else "Low Risk"

        # Display Results
        st.markdown("---")
        st.markdown("## Prediction Results")

        if prediction == 1:
            st.error(f"""
### At Risk
**Confidence: {confidence:.2f}%**

The model indicates an elevated risk of heart disease. Please consult a healthcare professional for proper evaluation.
            """)
        else:
            st.success(f"""
### Low Risk
**Confidence: {confidence:.2f}%**

The model indicates a lower risk of heart disease. Continue maintaining a healthy lifestyle and regular check-ups.
            """)

        # Show input summary
        with st.expander("View Input Summary", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information:**")
                st.write(f"- Age: {age} years")
                st.write(f"- Sex: {'Male' if sex == 1 else 'Female'}")
                st.write(f"- Chest Pain Type: {['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'][cp]}")
                st.write(f"- Resting BP: {trestbps} mm Hg")
                st.write(f"- Cholesterol: {chol} mg/dl")
                st.write(f"- Fasting Blood Sugar > 120: {'Yes' if fbs == 1 else 'No'}")
            
            with col2:
                st.markdown("**Advanced Metrics:**")
                st.write(f"- Resting ECG: {['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][restecg]}")
                st.write(f"- Max Heart Rate: {thalach} bpm")
                st.write(f"- Exercise Angina: {'Yes' if exang == 1 else 'No'}")
                st.write(f"- ST Depression: {oldpeak}")
                st.write(f"- Slope: {['Upsloping', 'Flat', 'Downsloping'][slope]}")
                st.write(f"- Major Vessels: {ca}")
                st.write(f"- Thalassemia: {['Normal', 'Fixed Defect', 'Reversible Defect'][thal-1]}")

        # Recommendations
        st.markdown("---")
        st.info("""
**Next Steps:**

- Save or screenshot your results for reference
- Discuss these findings with your healthcare provider
- Schedule regular health check-ups
- Maintain a heart-healthy lifestyle
- Monitor your vital signs regularly
        """)

# ---------------- MAIN APP ----------------
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Go to:",
        ["Home", "Prediction Tool"],
        index=0 if st.session_state.page == 'home' else 1
    )

    # Update session state based on selection
    if "Home" in page:
        st.session_state.page = 'home'
    else:
        st.session_state.page = 'prediction'

    # Sidebar Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This tool uses machine learning to predict heart disease risk based on medical parameters.
    
    **Version:** 1.0  
    **Model:** Logistic Regression  
    **Accuracy:** ~85%
    """)

    st.sidebar.markdown("---")
    st.sidebar.warning("Not a medical diagnosis. Consult a healthcare professional.")

    # Load model
    model, scaler = load_model()

    # Route to appropriate page
    if st.session_state.page == 'home':
        show_home_page()
    else:
        show_prediction_page(model, scaler)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p>© 2025 Heart Health AI </p>
        <p style="font-size: 0.9rem;">Not a substitute for professional medical advice</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()