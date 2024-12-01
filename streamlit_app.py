import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="ML Model Predictor",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
.stApp {
    background-color: #3D3F43FF;
}
.stButton>button {
    color: white;
    background-color: #4CAF50;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Title with Icon
st.title('🧠 ML Model Prediction & Retraining Platform')
st.write("A comprehensive tool for machine learning model interaction and management.")

# Sidebar for additional information
with st.sidebar:
    st.header("📊 Model Information")
    st.info("""
    This platform allows you to:
    - Make predictions using your trained model
    - Retrain the model with new data
    - Visualize prediction results
    
    💡 Tip: Ensure your input data matches the model's expected format.
    """)

# Prediction Section
st.header("🔮 Model Prediction")
st.subheader("Enter feature data to get predictions from the ML model")

col1, col2 = st.columns(2)

with col1:
    # Prediction Input
    input_data_str = st.text_input(
        '📥 Enter data for prediction', 
        '1.0,2.0,3.0,0.0,0.0,0.0,0.0', 
        help="Comma-separated feature values (minimum 21 features, pad with zeros if needed)"
    )

    try:
        # Process the input data
        input_data = [float(i.strip()) for i in input_data_str.split(',')]
        
        # Ensure input_data length is at least 21
        if len(input_data) < 21:
            input_data.extend([0.0] * (21 - len(input_data)))
        elif len(input_data) > 21:
            input_data = input_data[:21]

        if st.button('🚀 Generate Prediction', key='predict_btn'):
            with st.spinner('Processing your prediction...'):
                response = requests.post('http://localhost:8000/predict', json={'data': input_data})

                if response.status_code == 200:
                    st.write("Debug API Response:", response.json())
                    predictions = response.json().get('predictions', [])

                    # Check if the model returns a single prediction
                    if isinstance(predictions, list) and len(predictions) == 1 and isinstance(predictions[0], list):
                        single_prediction = predictions[0][0]
                        st.success(f"Prediction Result: {single_prediction}")

                        # Visualize the prediction as a single value
                        st.metric(label="Predicted Value", value=round(single_prediction, 3))
                    
                    # Handle multiple predictions (if applicable)
                    elif len(predictions) == len(input_data):
                        # Create DataFrame for visualization
                        pred_df = pd.DataFrame({
                            'Feature': [f'Feature {i+1}' for i in range(len(input_data))],
                            'Input Value': input_data,
                            'Prediction': predictions
                        })

                        # Create a grouped bar chart to compare input values and predictions
                        fig = px.bar(
                            pred_df.melt(id_vars=['Feature'], value_vars=['Input Value', 'Prediction']),
                            x='Feature',
                            y='value',
                            color='variable',
                            barmode='group',
                            title='Input vs Prediction Visualization',
                            labels={'value': 'Value', 'variable': 'Type'},
                            template='plotly_white'
                        )
                        st.plotly_chart(fig)
                    
                    else:
                        st.error("Unexpected prediction format returned by the API.")

                else:
                    st.error(f"Error in prediction: {response.text}")

    except ValueError:
        st.warning("❌ Please enter valid numeric values separated by commas")

with col2:
    st.info("""
    ### 🎯 Prediction Guidelines
    - Enter feature values as comma-separated numbers
    - Ensure the number of features matches model input
    - Example: `2,3,4,5,1,2,3,4,5,6,1,3,2,3,4,3,1,3,1,0,1`
    
    ### 📊 Interpretation
    The model will:
    - Process your input features
    - Generate predictions
    - Visualize results
    """)

# Retraining Section
st.header("🔄 Model Retraining")
st.subheader("Provide new training data to improve model performance")

col3, col4 = st.columns(2)

with col3:
    # Retraining Input
    retrain_data_str = st.text_area(
        '📊 Enter data for retraining', 
        '1.0,2.0,3.0\n4.0,5.0,6.0', 
        help="Each line represents a training sample, comma-separated"
    )
    
    retrain_labels_str = st.text_input(
        '🏷️ Enter corresponding labels', 
        '0,1', 
        help="Labels corresponding to each training sample"
    )

    if st.button('🔬 Retrain Model', key='retrain_btn'):
        try:
            # Parse training data
            retrain_data = [[float(i.strip()) for i in row.split(',')] for row in retrain_data_str.split('\n') if row]
            retrain_labels = [float(i.strip()) for i in retrain_labels_str.split(',') if i]

            # Validation: Check if the number of samples matches the number of labels
            if len(retrain_data) != len(retrain_labels):
                st.warning("❌ The number of training samples must match the number of labels.")
            else:
                with st.spinner('Retraining in progress...'):
                    response = requests.post('http://localhost:8000/retrain', json={
                        'data': retrain_data, 
                        'labels': retrain_labels
                    })
                    
                    if response.status_code == 200:
                        st.success("Model retrained successfully! 🎉")
                    else:
                        st.error(f"Retraining failed: {response.text}")
        
        except ValueError:
            st.warning("❌ Please enter valid numeric values for data and labels")

with col4:
    st.info("""
    ### 🔬 Retraining Best Practices
    - Provide multiple training samples
    - Match labels with corresponding data rows
    - Example Data: 
      ```
      1.0, 2.0, 3.0
      4.0, 5.0, 6.0
      ```
    - Example Labels: `0, 1`
    
    ### 💡 Tips
    - Ensure data diversity
    - Add informative samples
    - Monitor model performance
    """)

# Footer
st.markdown("---")
st.markdown("🤖 ML Model Interaction Platform | Powered by Streamlit & FastAPI")
st.markdown("© 2024 Smart Israel. All rights reserved.")
