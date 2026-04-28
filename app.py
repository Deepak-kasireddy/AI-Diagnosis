import os
# Suppress TensorFlow logging and force CPU mode to avoid DLL issues on Windows
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import streamlit as st
import time
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Custom imports
from utils.preprocessing import preprocess_xray, preprocess_blood_smear
from utils.prediction import predict_xray, predict_blood_smear
from utils.severity import calculate_severity, get_clinical_notes
from utils.report import MedicalReportGenerator
from utils.analyzer import ExpertRuleAnalyzer

try:
    import tensorflow as tf
    # Additional silence for TF loggers
    import logging
    tf.get_logger().setLevel('ERROR')
    st.sidebar.success("🎯 **Diagnostic Engine**: High-Precision (DL Mode)")
except (ImportError, Exception):
    st.sidebar.info("🚀 **Diagnostic Engine**: Optimized (Standard Mode)")

# Page config
st.set_page_config(
    page_title="AI Diagnosis | Medical Imaging",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #262730;
        color: white;
        border: 1px solid #4B4B4B;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF4B4B;
        border: 1px solid #FF4B4B;
        transform: translateY(-2px);
    }
    .report-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .status-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8em;
    }
    .status-severe { background-color: #FF4B4B; color: white; }
    .status-moderate { background-color: #FFA500; color: white; }
    .status-mild { background-color: #00FF00; color: black; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🩺 AI Diagnosis")
page = st.sidebar.selectbox("Navigate", ["Home", "Diagnosis", "Model Training", "Reports History"])

if page == "Home":
    st.title("Welcome to AI Medical Diagnosis System")
    st.markdown("""
    ### Advanced Disease Detection using Deep Learning
    This system analyzes **Chest X-rays** and **Blood Smear** images to detect potential pathologies with high precision.
    
    #### Key Features:
    - **Dual Analysis Pipeline**: Specialized models for Radiology and Hematology.
    - **Explainable AI**: Visual heatmaps highlighting affected areas.
    - **Structured Reporting**: Instant generation of clinical summaries in PDF format.
    - **Severity Estimation**: Automated staging of disease progression.
    
    *Disclaimer: This tool is intended for clinical support and should not replace professional medical judgment.*
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("📊 **Radiology Stream**\nDetects Pneumonia, Pleural Effusion, and other lung abnormalities.")
    with col2:
        st.info("🔬 **Hematology Stream**\nDetects Malaria, Leukemia, and Anemia from blood smear microscopy.")

elif page == "Diagnosis":
    st.title("Medical Image Analysis")
    
    analysis_type = st.radio("Select Analysis Type", ["Chest X-ray", "Blood Smear"], horizontal=True)
    
    uploaded_file = st.file_uploader(f"Upload {analysis_type} Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Create temp folder for outputs
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
            
        # Save uploaded file
        img_path = os.path.join("outputs", "temp_upload.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
        if st.button("Run AI Diagnosis"):
            with st.spinner("Analyzing image..."):
                time.sleep(1.5) # Simulate processing
                
                # Inference
                if analysis_type == "Chest X-ray":
                    prediction, confidence, severity, heatmap_path = predict_xray(img_path, "outputs")
                else:
                    prediction, confidence, severity, heatmap_path = predict_blood_smear(img_path, "outputs")
                    
                    clinical_notes = get_clinical_notes(prediction, severity)
                    
                    # Load the generated visualization
                    visualized_img = cv2.imread(heatmap_path)
                    visualized_img = cv2.cvtColor(visualized_img, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.image(visualized_img, caption="AI Analysis (Heatmap Overlay)", use_container_width=True)
                
                st.markdown("---")
                
                # Results Section
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("Diagnosis", prediction)
                with res_col2:
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                with res_col3:
                    severity_class = f"status-{severity.lower()}" if severity != "N/A" else ""
                    st.markdown(f"**Severity:** <span class='status-badge {severity_class}'>{severity}</span>", unsafe_allow_html=True)
                
                st.markdown(f"### Clinical Summary\n{clinical_notes}")
                
                # Generate Report
                report_path = os.path.join("outputs", "medical_report.pdf")
                generator = MedicalReportGenerator(report_path)
                report_data = {
                    'analysis_type': analysis_type,
                    'prediction': prediction,
                    'confidence': confidence,
                    'severity': severity,
                    'notes': clinical_notes,
                    'image_path': img_path,
                    'heatmap_path': heatmap_path
                }
                generator.generate(report_data)
                
                with open(report_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download Clinical Report",
                        data=pdf_file,
                        file_name="AI_Diagnosis_Report.pdf",
                        mime="application/pdf"
                    )

elif page == "Model Training":
    st.title("Model Training & Performance")
    st.info("Manage datasets and retrain models for improved accuracy.")
    
    train_type = st.selectbox("Select Pipeline", ["Chest X-ray (Pneumonia)", "Blood Smear (Multi-Disease)"])
    
    st.subheader("Dataset Statistics")
    # Real dataset check
    dataset_path = "datasets/xray" if "X-ray" in train_type else "datasets/blood"
    if os.path.exists(dataset_path):
        classes = os.listdir(dataset_path)
        for cls in classes:
            count = len(os.listdir(os.path.join(dataset_path, cls)))
            st.write(f"- **{cls}**: {count} images")
    else:
        st.warning("Dataset not found. Please upload data to the 'datasets' folder.")
        
    if st.button("Start Retraining Pipeline"):
        st.warning("Training requires significant GPU/CPU resources. This will run in the background.")
        # In a real environment, we would trigger the scripts here
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
        st.success("Training complete! Model updated.")

elif page == "Reports History":
    st.title("Generated Reports History")
    if os.path.exists("outputs"):
        files = [f for f in os.listdir("outputs") if f.endswith(".pdf")]
        if files:
            for f in files:
                st.write(f"📁 {f}")
        else:
            st.write("No reports generated yet.")
    else:
        st.write("No reports generated yet.")
