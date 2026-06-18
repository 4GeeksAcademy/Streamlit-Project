

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Student Success Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Automated Training & Model Loading Pipeline ---
@st.cache_resource
def train_and_load_model():
    dataset_filename = 'student-por.csv'
    model_filename = 'model.pkl'
    
    if os.path.exists(model_filename):
        try:
            return joblib.load(model_filename)
        except Exception:
            pass

    if not os.path.exists(dataset_filename):
        st.error(f"⚠️ Dataset '{dataset_filename}' not found in the directory. Please upload it.")
        return None
        
    try:
        df = pd.read_csv(dataset_filename, sep=';')
        
        # Target variable pass_status (1 if G3 >= 10 (which is >= 50%), else 0)
        df['pass_status'] = (df['G3'] >= 10).astype(int)
        
        features = [
            'age', 'studytime', 'failures', 'absences', 'G1', 'G2', 
            'schoolsup', 'famsup', 'paid', 'internet', 'higher', 'romantic'
        ]
        
        X = df[features].copy()
        y = df['pass_status']
        
        for col in ['schoolsup', 'famsup', 'paid', 'internet', 'higher', 'romantic']:
            X[col] = X[col].map({'yes': 1, 'no': 0}).fillna(0)
            
        X = X.fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        joblib.dump(model, model_filename)
        return model
        
    except Exception as e:
        st.error(f"Error executing automated training pipeline: {e}")
        return None

model = train_and_load_model()

# --- 3. Dashboard Header & Layout ---
st.title("🎓 Student Success Analytics Platform")
st.write("Predict whether a student is likely to pass based on academic, demographic, and school-related features.")

left_column, right_column = st.columns([1, 1])

with left_column:
    st.subheader("📝 Student Information Form")
    
    with st.form("student_form"):
        age = st.slider("Age", min_value=15, max_value=22, value=17)
        studytime = st.selectbox("Study Time (hours/week)", options=[1, 2, 3, 4], format_func=lambda x: ["< 2 hours", "2 to 5 hours", "5 to 10 hours", "> 10 hours"][x-1])
        failures = st.number_input("Past Class Failures", min_value=0, max_value=5, value=0, step=1)
        absences = st.number_input("School Absences", min_value=0, max_value=93, value=0, step=1)
        
        st.divider()
        st.markdown("##### Academic Grades (Percentage System)")
        # Sliders set to 0-100%
        g1_pct = st.slider("First Period Grade (G1) %", min_value=0, max_value=100, value=50)
        g2_pct = st.slider("Second Period Grade (G2) %", min_value=0, max_value=100, value=50)
        
        st.divider()
        st.markdown("##### Support & Environment")
        schoolsup = st.selectbox("Extra Educational Support", ["no", "yes"])
        famsup = st.selectbox("Family Educational Support", ["no", "yes"])
        paid = st.selectbox("Paid Extra Classes (Within Course)", ["no", "yes"])
        internet = st.selectbox("Internet Access at Home", ["no", "yes"])
        higher = st.selectbox("Wants to Take Higher Education", ["no", "yes"])
        romantic = st.selectbox("With a Romantic Relationship", ["no", "yes"])
        
        submit_button = st.form_submit_button(label="Run Prediction")

with right_column:
    st.subheader("📊 Prediction & Insights")
    
    if submit_button:
        # Convert 0-100% back to 0-20 scale for the model (e.g., 50% -> 10)
        g1_mapped = (g1_pct / 100) * 20
        g2_mapped = (g2_pct / 100) * 20
        
        input_dict = {
            'age': age,
            'studytime': studytime,
            'failures': failures,
            'absences': absences,
            'G1': g1_mapped,
            'G2': g2_mapped,
            'schoolsup': 1 if schoolsup == "yes" else 0,
            'famsup': 1 if famsup == "yes" else 0,
            'paid': 1 if paid == "yes" else 0,
            'internet': 1 if internet == "yes" else 0,
            'higher': 1 if higher == "yes" else 0,
            'romantic': 1 if romantic == "yes" else 0,
        }
        
        input_df = pd.DataFrame([input_dict])
        
        with st.spinner("Analyzing academic trajectory..."):
            if model is not None:
                prediction = model.predict(input_df)
                pass_status = prediction[0]
                
                if pass_status == 1:
                    st.success("🎉 **Prediction: The student is LIKELY TO PASS.**")
                    st.balloons()
                    st.markdown("The academic indicators suggest steady performance. Maintain support structures to ensure success.")
                else:
                    st.error("⚠️ **Prediction: The student is AT RISK of failing.**")
                    st.markdown("Early intervention is recommended. Consider utilizing school support systems and tutoring resources.")
                
                # Visual gauge using the percentage average
                progress_val = int((g1_pct + g2_pct) / 2)
                st.progress(progress_val)
            else:
                st.warning("Model pipeline initialization failed. Check dataset path and logs.")
    else:
        st.info("Fill out the Student Information Form on the left and click 'Run Prediction' to see analytics.")

# --- 4. Sidebar Documentation & Configuration ---
st.sidebar.header("⚙️ Configuration & Resources")
st.sidebar.markdown("**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student%2Bperformance)")
st.sidebar.markdown("**Tech Stack:** Python, Pandas, Scikit-Learn, Streamlit, Render")
st.sidebar.markdown("---")
st.sidebar.markdown("**Documentation:**")
st.sidebar.markdown("- [Streamlit Reference](https://docs.streamlit.io/)")
st.sidebar.markdown("- [Scikit-Learn ML](https://scikit-learn.org/)")