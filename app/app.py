import sys
import os
import streamlit as st

# Add the project root directory to python path so we can import src.predict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import ChurnPredictor

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    layout="centered"
)

st.title("Telco Customer Churn Predictor")
st.write("Enter customer subscription details below to predict their likelihood of churning.")

# Initialize predictor
@st.cache_resource
def load_predictor():
    return ChurnPredictor(model_dir="models")

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Build the form inputs
with st.form("churn_form"):
    st.subheader("Customer Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        seniorcitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Services & Subscriptions")
    col4, col5 = st.columns(2)
    with col4:
        tenure = st.number_input("Tenure (Months)", min_value=1, max_value=100, value=12)
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
        multiplelines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        onlinesecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        onlinebackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    
    with col5:
        deviceprotection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        techsupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streamingtv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streamingmovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])

    paymentmethod = st.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    st.subheader("Billing Details")
    col6, col7 = st.columns(2)
    with col6:
        monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=1.0)
    with col7:
        totalcharges = st.number_input("Total Charges ($)", min_value=0.0, value=780.0, step=5.0)

    submit_button = st.form_submit_button("Predict Churn Risk", use_container_width=True)

# Process prediction when submitted
if submit_button:
    customer_data = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges
    }

    result = predictor.predict(customer_data)
    prob_percent = result["churn_probability"] * 100

    st.markdown("---")
    if result["prediction"] == "Churn":
        st.error(f"High Churn Risk!**\n\nEstimated Churn Probability: **{prob_percent:.1f}%")
    else:
        st.success(f"Customer Likely to Stay.**\n\nEstimated Churn Probability: **{prob_percent:.1f}%")