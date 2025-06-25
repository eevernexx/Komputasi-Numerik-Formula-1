# ========================================
# STREAMLIT APP - FORMULA 1 LAP TIME PREDICTOR
# Interactive Web Application
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ========================================
# PAGE CONFIG
# ========================================

st.set_page_config(
    page_title="🏁 F1 Lap Time Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# LOAD MODEL ARTIFACTS
# ========================================

@st.cache_resource
def load_model_artifacts():
    """Load semua model artifacts dengan caching"""
    try:
        # Load model
        with open('f1_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load encoders
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load metadata
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load sample data
        with open('sample_data.pkl', 'rb') as f:
            sample_data = pickle.load(f)
        
        return model, scaler, encoders, feature_names, metadata, sample_data
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, None

# Load artifacts
model, scaler, encoders, feature_names, metadata, sample_data = load_model_artifacts()

# ========================================
# TITLE & HEADER
# ========================================

st.title("🏁 Formula 1 Lap Time Predictor")
st.markdown("### Predict F1 lap times using machine learning")

# Display model info
if metadata:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", metadata['model_type'])
    
    with col2:
        st.metric("Test RMSE", f"{metadata['test_rmse']:.3f}s")
    
    with col3:
        st.metric("R² Score", f"{metadata['test_r2']:.3f}")
    
    with col4:
        st.metric("Features", metadata['feature_count'])

st.markdown("---")

# ========================================
# SIDEBAR - INPUT FEATURES
# ========================================

st.sidebar.header("🎯 Input Features")
st.sidebar.markdown("Adjust the parameters below to predict lap time:")

# Feature inputs
def create_feature_inputs():
    """Create input widgets untuk features"""
    
    inputs = {}
    
    # Basic race info
    st.sidebar.subheader("🏁 Race Information")
    inputs['year'] = st.sidebar.slider("Year", 1950, 2024, 2023)
    inputs['round'] = st.sidebar.slider("Round", 1, 24, 10)
    inputs['lap'] = st.sidebar.slider("Lap Number", 1, 70, 25)
    
    # Driver info
    st.sidebar.subheader("👤 Driver Information")
    inputs['driver_age'] = st.sidebar.slider("Driver Age", 18, 50, 28)
    inputs['driver_experience'] = st.sidebar.slider("Driver Experience (years)", 0, 25, 5)
    
    # Circuit info  
    st.sidebar.subheader("🏁 Circuit Information")
    circuit_difficulty_input = st.sidebar.slider("Circuit Difficulty", 1, 100, 50)
    inputs['circuit_difficulty'] = (circuit_difficulty_input - 1) / 99  # Convert to 0-1 range
    inputs['circuit_alt'] = st.sidebar.slider("Circuit Altitude (m)", -50, 3500, 500)
    
    # Constructor info
    st.sidebar.subheader("🏎️ Constructor Information")
    inputs['constructor_avg_points'] = st.sidebar.slider("Constructor Avg Points", 0.0, 25.0, 10.0)
    
    # Race context
    st.sidebar.subheader("📊 Race Context")
    inputs['position'] = st.sidebar.slider("Current Position", 1, 20, 8)
    season_progress_input = st.sidebar.slider("Season Progress (%)", 1, 100, 50)
    inputs['season_progress'] = season_progress_input / 100  # Convert to 0-1 range
    
    # Era (simplified)
    era_options = ['Classic', 'Turbo', 'Refuel', 'V8', 'Hybrid']
    selected_era = st.sidebar.selectbox("F1 Era", era_options, index=4)
    
    # Convert era to numeric (dummy encoding)
    for era in era_options:
        inputs[f'f1_era_{era}'] = 1 if era == selected_era else 0
    
    # Nationality (simplified)
    nationality_options = ['British', 'German', 'Italian', 'French', 'Spanish', 'Other']
    selected_nat = st.sidebar.selectbox("Driver Nationality", nationality_options, index=0)
    
    # Convert nationality to numeric
    for nat in nationality_options:
        inputs[f'nationality_{nat}'] = 1 if nat == selected_nat else 0
    
    # Lap context (radio buttons)
    lap_phase = st.sidebar.radio("Lap Phase", ['Early Race', 'Mid Race', 'Late Race'])
    inputs['early_race'] = 1 if lap_phase == 'Early Race' else 0
    inputs['mid_race'] = 1 if lap_phase == 'Mid Race' else 0 
    inputs['late_race'] = 1 if lap_phase == 'Late Race' else 0
    
    # Position context
    position_context = st.sidebar.radio("Position Context", ['Front Runner', 'Midfield', 'Back Marker'])
    inputs['front_runner'] = 1 if position_context == 'Front Runner' else 0
    inputs['midfield'] = 1 if position_context == 'Midfield' else 0
    inputs['back_marker'] = 1 if position_context == 'Back Marker' else 0
    
    return inputs

# ========================================
# MAIN APP LAYOUT
# ========================================

if model is not None:
    
    # Create two main columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔮 Lap Time Prediction")
        
        # Get user inputs
        user_inputs = create_feature_inputs()
        
        # Predict button
        if st.button("🚀 Predict Lap Time", type="primary"):
            
            with st.spinner("Predicting lap time..."):
                time.sleep(1)  # Dramatic effect
                
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([user_inputs])
                    
                    # Add missing features with defaults
                    for feature in feature_names:
                        if feature not in input_df.columns:
                            input_df[feature] = 0  # Default value
                    
                    # Reorder columns to match training
                    input_df = input_df.reindex(columns=feature_names, fill_value=0)
                    
                    # Scale features
                    input_scaled = scaler.transform(input_df)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    
                    # Display result
                    st.success("✅ Prediction Complete!")
                    
                    # Big prediction display
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;">
                        <h1 style="color: #1f77b4; margin: 0;">⏱️ {prediction:.3f} seconds</h1>
                        <h3 style="color: #666; margin: 10px 0;">Predicted Lap Time</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interpretation
                    if prediction < 80:
                        interpretation = "🚀 Very Fast Lap! (Street circuit or qualifying pace)"
                        color = "green"
                    elif prediction < 100:
                        interpretation = "⚡ Fast Lap! (Typical modern F1 circuit)"
                        color = "blue"
                    elif prediction < 120:
                        interpretation = "🏁 Normal Lap Time (Standard race pace)"
                        color = "orange"
                    else:
                        interpretation = "🐌 Slow Lap (Long circuit or difficult conditions)"
                        color = "red"
                    
                    st.markdown(f"**{interpretation}**")
                    
                    # Additional insights
                    st.info(f"""
                    💡 **Insights:**
                    - For comparison, Monaco GP: ~75-80s, Spa-Francorchamps: ~105-110s
                    - Current position: P{user_inputs['position']} influences strategy
                    - {selected_era} era technology affects performance
                    - Circuit difficulty: {circuit_difficulty_input}/100 (higher = more challenging)
                    - Season progress: {season_progress_input}% complete
                    """)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    with col2:
        st.subheader("📊 Model Information")
        
        # Model performance
        st.markdown("**🎯 Model Performance:**")
        st.write(f"• Test RMSE: {metadata['test_rmse']:.3f}s")
        st.write(f"• R² Score: {metadata['test_r2']:.3f}")
        st.write(f"• Accuracy: {(1-metadata['test_rmse']/100)*100:.1f}%")
        
        # Dataset info
        st.markdown("**📊 Dataset:**")
        st.write(f"• Total samples: {metadata['dataset_info']['total_samples']:,}")
        st.write(f"• Training: {metadata['dataset_info']['training_samples']:,}")
        st.write(f"• Features: {metadata['feature_count']}")
        
        # Quick examples
        st.markdown("**⚡ Quick Examples:**")
        
        example_scenarios = [
            {"name": "🏆 Pole Position", "values": {"position": 1, "driver_experience": 10, "constructor_avg_points": 20}},
            {"name": "🏁 Midfield Battle", "values": {"position": 8, "driver_experience": 3, "constructor_avg_points": 5}},
            {"name": "🐌 Back Marker", "values": {"position": 18, "driver_experience": 1, "constructor_avg_points": 1}}
        ]
        
        for scenario in example_scenarios:
            if st.button(scenario["name"], key=scenario["name"]):
                st.session_state.update(scenario["values"])
                st.experimental_rerun()

    # ========================================
    # ADDITIONAL FEATURES
    # ========================================
    
    st.markdown("---")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    title="Top 10 Most Important Features")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample predictions visualization
    if st.checkbox("📈 Show Sample Predictions"):
        st.subheader("📊 Sample Predictions Comparison")
        
        if sample_data is not None:
            # Get sample predictions
            sample_predictions = model.predict(sample_data.head(20))
            
            # Create comparison chart
            comparison_df = pd.DataFrame({
                'Sample': range(1, 21),
                'Predicted_Time': sample_predictions
            })
            
            fig = px.line(comparison_df, x='Sample', y='Predicted_Time',
                         title="Sample Lap Time Predictions",
                         labels={'Predicted_Time': 'Lap Time (seconds)'})
            
            fig.update_traces(mode='markers+lines')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Model artifacts not found! Please ensure all model files are in the same directory.")
    st.info("Required files: f1_model.pkl, scaler.pkl, encoders.pkl, feature_names.pkl, model_metadata.pkl")

# ========================================
# FOOTER
# ========================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🏁 Formula 1 Lap Time Predictor | Built with Streamlit & Machine Learning<br>
    Data: Formula 1 Historical Dataset (1950-2024) | Model: {model_type}
</div>
""".format(model_type=metadata['model_type'] if metadata else 'ML Model'), unsafe_allow_html=True)
