import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

    # The label mapping is hardcoded since the original file may not contain the label column
    nutrient_gap_labels = {
        0: 'Small Nutrient Gap',
        1: 'Significant Nutrient Gap',
        2: 'Severe Nutrient Gap'
    }

except FileNotFoundError as e:
    st.error(f"Error: Required data files not found. {e}. Please ensure the files are in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during setup: {e}")
    st.stop()

# --- Preprocessing Pipeline ---
# Recreate the preprocessing pipeline and fit it
num_pipeline = Pipeline([
    ("scalar", StandardScaler()),
    ("variance", VarianceThreshold(threshold=0.1)),
    ("selector", SelectKBest(score_func=f_classif))
])

# Define features and target for pipeline fitting
X_for_pipeline_fit = nutrient_gap_encoded.drop(columns=["nutrient_gap_level_encoded", "mnari", "nutrient_gap_level"], errors='ignore')
y_for_pipeline_fit = nutrient_gap_encoded["nutrient_gap_level_encoded"]
num_pipeline.fit(X_for_pipeline_fit, y_for_pipeline_fit)
selected_features_names = X_for_pipeline_fit.columns[num_pipeline.named_steps["selector"].get_support()]

# --- Utility Function for Prediction ---

def get_prediction_result(input_df_full, pipeline, model, labels):
    """
    Transforms the input data, makes a prediction, and returns the result.
    """
    try:
        # Transform the data using the fitted pipeline
        processed_data = pipeline.transform(input_df_full.copy())
        
        # Make a prediction
        predicted_level_encoded = model.predict(processed_data)[0]
        
        # Map the prediction to a human-readable label
        predicted_level_label = labels.get(int(predicted_level_encoded), 'Unknown')
        
        return predicted_level_label
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- Streamlit UI ---

st.title("Nutrient Gap Prediction & Intervention Simulation")

st.header("User Input Guide")
st.write("Please use the following numerical codes for your input:")

# Create a mapping from category code to region name for display
category_region_map = nutrient_gap_original[['category', 'region']].drop_duplicates().set_index('category')['region'].to_dict()
region_options = sorted(list(category_region_map.keys()))

st.header("Nutrient Gap Prediction")
st.write("Select a region to get the initial nutrient gap prediction.")

# Create the input fields dynamically, now handling only category
initial_input_data = {}
selected_category = st.selectbox("Region (Category)", options=region_options, format_func=lambda x: category_region_map.get(x, x))
initial_input_data['category'] = selected_category


if st.button("Get Initial Prediction"):
    try:
        # Create a dataframe with all features and set their values
        # All other features are defaulted to 0
        input_df_full = pd.DataFrame(np.zeros((1, X_for_pipeline_fit.shape[1])), columns=X_for_pipeline_fit.columns)
        
        # Set the category feature
        if 'category' in input_df_full.columns:
            input_df_full['category'] = initial_input_data['category']

        st.subheader("Initial Input Data")
        # Display all features that are used by the model
        st.dataframe(input_df_full[selected_features_names])

        initial_predicted_level_label = get_prediction_result(input_df_full, num_pipeline, model, nutrient_gap_labels)
        selected_region_name = category_region_map.get(initial_input_data['category'], 'Unknown Region')
        
        if initial_predicted_level_label:
            st.subheader(f"Initial Predicted Nutrient Gap Level in {selected_region_name}: {initial_predicted_level_label}")

            initial_prediction_data = {'Nutrient Gap Level': [initial_predicted_level_label], 'Value': [1]}
            initial_prediction_df = pd.DataFrame(initial_prediction_data)
            plt.figure(figsize=(6, 4))
            sns.barplot(data=initial_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
            plt.title(f"Initial Predicted Nutrient Gap Level in {selected_region_name}")
            plt.ylabel("")
            plt.yticks([])
            st.pyplot(plt)
            plt.clf()

            if initial_predicted_level_label == 'Small Nutrient Gap':
                st.success("The initial prediction indicates a small nutrient gap. Continue monitoring and promoting healthy diets.")
            elif initial_predicted_level_label == 'Significant Nutrient Gap':
                st.warning("The initial prediction indicates a significant nutrient gap. Consider implementing targeted interventions to address this.")
            else:
                st.error("The initial prediction indicates a severe nutrient gap. Urgent and comprehensive interventions are highly recommended.")

            st.session_state['initial_input_df_full'] = input_df_full
    except Exception as e:
        st.error(f"Error during initial prediction: {e}")

st.header("Intervention Simulation")
st.write("Select an intervention scenario to see its potential impact on the nutrient gap level.")

intervention_scenarios = {
    "No Intervention": {},
    "Increase in Nutritious Food": {
        'avg_ca(mg)': 1.1,
        'avg_thiamin(mg)': 1.1,
        'avg_vitb12(mcg)': 1.1,
        'pcfci': 0.9,
        'avg_vita(mcg)': 1.1,
        'avg_riboflavin(mg)': 1.1,
        'avg_niacin(mg)': 1.1,
        'millet(mt)': 1.2,
        'sorghum(mt)': 1.2
    },
    "Promote Fortified Foods": {
        'avg_ca(mg)': 1.1,
        'avg_thiamin(mg)': 1.1,
        'avg_vitb12(mcg)': 1.1,
        'pcfci': 0.9,
        'avg_vita(mcg)': 1.1,
        'avg_riboflavin(mg)': 1.1,
        'avg_niacin(mg)': 1.1
    },
    "Improve Climate Resilience": {
        'pcfci': 0.9,
        'sorghum(mt)': 1.3,
        'millet(mt)': 1.3
    }
}

selected_scenario = st.selectbox("Select an intervention scenario:", list(intervention_scenarios.keys()))

if st.button("Simulate Intervention"):
    if 'initial_input_df_full' in st.session_state:
        try:
            # Display the initial data for comparison first
            st.subheader("Initial Data (For Comparison)")
            initial_df_to_display = st.session_state['initial_input_df_full'][selected_features_names]
            st.dataframe(initial_df_to_display)
            
            simulated_df_full = st.session_state['initial_input_df_full'].copy()
            scenario_impact = intervention_scenarios[selected_scenario]

            for feature, multiplier in scenario_impact.items():
                if feature in simulated_df_full.columns:
                    simulated_df_full[feature] *= multiplier
                else:
                    st.write(f"Warning: Scenario impacts feature '{feature}' not found in the simulation dataframe.")
            
            st.subheader(f"Simulated Data after {selected_scenario} Intervention")
            # Display only the selected features from the simulated data
            simulated_df_to_display = simulated_df_full[selected_features_names]
            st.dataframe(simulated_df_to_display)

            predicted_level_label = get_prediction_result(simulated_df_full, num_pipeline, model, nutrient_gap_labels)
            
            # Get the region name from the initial state
            initial_category = st.session_state['initial_input_df_full']['category'].iloc[0]
            selected_region_name = category_region_map.get(initial_category, 'Unknown Region')

            if predicted_level_label:
                st.subheader(f"Predicted Nutrient Gap Level in {selected_region_name} after {selected_scenario}: {predicted_level_label}")

                simulated_prediction_data = {'Nutrient Gap Level': [predicted_level_label], 'Value': [1]}
                simulated_prediction_df = pd.DataFrame(simulated_prediction_data)

                plt.figure(figsize=(6, 4))
                sns.barplot(data=simulated_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
                plt.title(f"Predicted Nutrient Gap Level after {selected_scenario}")
                plt.ylabel("")
                plt.yticks([])
                st.pyplot(plt)
                plt.clf()

                if predicted_level_label == 'Small Nutrient Gap':
                    st.success(f"Based on the '{selected_scenario}' intervention, the predicted nutrient gap is small.")
                elif predicted_level_label == 'Significant Nutrient Gap':
                    st.warning(f"Based on the '{selected_scenario}' intervention, the predicted nutrient gap is significant. This area may require targeted interventions.")
                else:
                    st.error(f"Based on the '{selected_scenario}' intervention, the predicted nutrient gap is severe. Urgent intervention may be needed in this area.")

        except Exception as e:
            st.error(f"Error during simulation: {e}")
    else:
        st.warning("Please get the initial prediction first before simulating interventions.")
