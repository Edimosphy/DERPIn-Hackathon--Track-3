import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

# --- Global Variables and Data Loading ---
try:
    # Load data and model
    nutrient_gap_original = pd.read_csv('nutrient_gap.csv')
    nutrient_gap_encoded = pd.read_csv('nutrient_gap_encoded.csv')
    
    # Ensure encoded target column is integer type
    if 'nutrient_gap_level_encoded' in nutrient_gap_encoded.columns:
        nutrient_gap_encoded['nutrient_gap_level_encoded'] = nutrient_gap_encoded['nutrient_gap_level_encoded'].astype(int)

    # Load the trained model
    model = joblib.load('nutrient_gap_model.pkl')

    # Create the label encoder and mapping globally
    label_encoder = LabelEncoder()
    label_encoder.fit(nutrient_gap_original['nutrient_gap_level'])
    nutrient_gap_labels = {
        0: 'Small Nutrient Gap',
        1: 'Significant Nutrient Gap',
        2: 'Severe Nutrient Gap'
    }

except FileNotFoundError as e:
    st.error(f"Error: Required data files not found. {e}. Please ensure the files are in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data/model loading: {e}")
    st.stop()

# --- Utility Functions ---
def get_prediction_result(df, pipeline, model, labels):
    """
    Makes a prediction using the trained pipeline and model.
    Returns the predicted label string or None if an error occurs.
    """
    try:
        # Preprocess the input dataframe using the pipeline
        processed_data = pipeline.transform(df)
        # Make the prediction
        prediction = model.predict(processed_data)
        # Get the predicted class label from the mapping
        predicted_level = labels.get(prediction[0])
        return predicted_level
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# --- Main Streamlit App Logic ---
st.title("Nutrient Gap Prediction and Simulation")

st.header("1. Initial Prediction")

# Create a list of regions for the dropdown
regions = sorted(nutrient_gap_original['region'].unique())
selected_region = st.selectbox("Select a Region", regions)

if st.button("Get Initial Prediction"):
    if 'model' in locals():
        st.spinner("Predicting initial nutrient gap level...")
        try:
            # Filter data for the selected region
            region_data = nutrient_gap_encoded[nutrient_gap_encoded['region'] == selected_region].copy()
            
            # Drop the 'region' and target columns for prediction
            features = region_data.drop(columns=['region', 'nutrient_gap_level_encoded', 'category'], errors='ignore')

            # Create the preprocessing pipeline (assuming it's not saved)
            # This is a simplified version, as the real pipeline would need to be loaded
            # or recreated exactly as it was during training.
            num_features = features.select_dtypes(include=np.number).columns.tolist()
            num_pipeline = Pipeline([
                ('variance', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler())
            ])
            
            st.session_state['initial_prediction_df'] = features
            st.session_state['num_pipeline'] = num_pipeline

            # Get the prediction result
            predicted_level_label = get_prediction_result(features, num_pipeline, model, nutrient_gap_labels)
            
            if predicted_level_label:
                st.subheader("Initial Nutrient Gap Level")
                st.success(f"For {selected_region}, the initial nutrient gap is predicted to be: **{predicted_level_label}**")

                # Display a simple bar chart of the initial prediction
                initial_prediction_df = pd.DataFrame({'Nutrient Gap Level': [predicted_level_label], 'Value': [1]})
                plt.figure(figsize=(6, 4))
                sns.barplot(data=initial_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
                plt.title("Initial Predicted Nutrient Gap Level")
                plt.ylabel("")
                plt.yticks([])
                st.pyplot(plt)
                plt.close()
            else:
                st.warning("Could not get a valid initial prediction. Please try again.")

        except Exception as e:
            st.error(f"Error during initial prediction: {e}")
    else:
        st.error("Model not loaded. Please check the file paths.")

# --- Simulation of Interventions ---
st.markdown("---")
st.header("2. Simulate Intervention")
st.write("Simulate the effect of a positive intervention by increasing commodity-related features.")

if 'initial_prediction_df' in st.session_state and 'num_pipeline' in st.session_state:
    st.write("Initial prediction is available. You can now run a simulation.")
    
    # Define a list of scenarios to apply
    intervention_options = {
        'Increase Agricultural Production': ['maize(mt)', 'rice(mt)', 'sorghum(mt)', 'cassava(mt)', 'millet(mt)'],
        'Improve Food Availability': ['price'] # Price decrease simulated as an increase in availability
    }
    
    selected_scenario = st.selectbox("Select a scenario to simulate:", list(intervention_options.keys()))

    # New user input for the percentage increase
    percentage_increase = st.slider("Percentage Increase (%)", 0, 100, 10, step=5)
    
    # Calculate the new multiplier based on the user's formula
    percentage_decimal = percentage_increase / 100
    multiplier = 1 + (percentage_decimal * 2)

    if st.button("Run Simulation"):
        try:
            # Retrieve the initial prediction dataframe and pipeline from session state
            initial_df = st.session_state['initial_prediction_df'].copy()
            num_pipeline = st.session_state['num_pipeline']
            
            # Get the features to be multiplied based on the selected scenario
            features_to_multiply = intervention_options[selected_scenario]

            # Create a copy of the dataframe to simulate the changes
            simulated_df_full = initial_df.copy()

            # Apply the multiplier to the selected features
            for feature in features_to_multiply:
                if feature in simulated_df_full.columns:
                    simulated_df_full[feature] *= multiplier

            # Get the new prediction with the simulated data
            predicted_level_label = get_prediction_result(simulated_df_full, num_pipeline, model, nutrient_gap_labels)
            
            if predicted_level_label:
                st.subheader(f"Predicted Nutrient Gap Level after {selected_scenario}: {predicted_level_label}")

                simulated_prediction_df = pd.DataFrame({'Nutrient Gap Level': [predicted_level_label], 'Value': [1]})
                plt.figure(figsize=(6, 4))
                sns.barplot(data=simulated_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
                plt.title(f"Predicted Nutrient Gap Level after {selected_scenario}")
                plt.ylabel("")
                plt.yticks([])
                st.pyplot(plt)
                plt.close()

                if predicted_level_label == 'Small Nutrient Gap':
                    st.success(f"Based on '{selected_scenario}', the predicted nutrient gap is small.")
                elif predicted_level_label == 'Significant Nutrient Gap':
                    st.warning(f"Based on '{selected_scenario}', the predicted nutrient gap is significant.")
                else:
                    st.error(f"Based on '{selected_scenario}', the predicted nutrient gap is severe.")
        except Exception as e:
            st.error(f"Error during simulation: {e}")
    else:
        st.warning("Please get the initial prediction first before simulating interventions.")
else:
    st.info("Please get an initial prediction first to enable the simulation feature.")
