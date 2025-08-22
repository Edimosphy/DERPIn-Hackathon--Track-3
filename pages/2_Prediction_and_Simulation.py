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
        # Use a copy to avoid a SettingWithCopyWarning
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
st.write("Input the corresponding numerical codes for your inputs.")

# PCFCI Code
st.write("pcfci code guide")
st.write("""Vulnerability Level: Use the slider to pick your range value, 
0.0: Vulnerable, 
1.0: Not Vulnerable
""")

# Create a mapping from category code to region name for display
category_region_map = nutrient_gap_original[['category', 'region']].drop_duplicates().set_index('category')['region'].to_dict()
region_options = sorted(list(category_region_map.keys()))

st.header("Nutrient Gap Prediction")
st.write("Enter the feature values below to get the initial nutrient gap prediction.")

# Dictionary to hold all user inputs
initial_input_data = {}

# Create a separate selection for region
selected_category = st.selectbox("Select a Region (Category)", options=region_options, format_func=lambda x: category_region_map.get(x, x))
initial_input_data['category'] = selected_category

# Create input fields for all selected features, except 'category'
input_cols_for_ui = [col for col in selected_features_names if col != 'category']

for feature in input_cols_for_ui:
    display_label = feature.replace('_', ' ').title()
    if feature in ['sorghum(mt)', 'millet(mt)']:
        min_val = 1.0
        max_val = 100000.00
        mean_val = float(X_for_pipeline_fit[feature].mean())
        default_val = max(min(mean_val, max_val), min_val)
        initial_input_data[feature] = st.slider(
            f"Select a value for {display_label}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=1.0
        )
    elif feature in X_for_pipeline_fit.columns:
        min_val = 0.0
        max_val = 100.0
        mean_val = float(X_for_pipeline_fit[feature].mean())
        default_val = max(min(mean_val, max_val), min_val)
        step_val = 1.0 if X_for_pipeline_fit[feature].nunique() <= 2 else 0.01
        initial_input_data[feature] = st.slider(
            f"Select a value for {display_label}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step_val
        )
    else:
        initial_input_data[feature] = 0.0

if st.button("Get Initial Prediction"):
    try:
        # Create a dataframe with all features and set their values
        input_df_full = pd.DataFrame(np.zeros((1, X_for_pipeline_fit.shape[1])), columns=X_for_pipeline_fit.columns)
        for feature, value in initial_input_data.items():
            if feature in input_df_full.columns:
                input_df_full[feature] = value

        # Get initial prediction and store it in session state
        initial_predicted_level_label = get_prediction_result(input_df_full, num_pipeline, model, nutrient_gap_labels)
        selected_region_name = category_region_map.get(input_df_full['category'].iloc[0], 'Unknown Region')

        st.session_state['initial_input_df_full'] = input_df_full
        st.session_state['initial_prediction_run'] = True
        st.session_state['initial_prediction_label'] = initial_predicted_level_label
        st.session_state['initial_region_name'] = selected_region_name

    except Exception as e:
        st.error(f"Error during initial prediction setup: {e}")

# --- Display Initial Prediction ---
if 'initial_prediction_run' in st.session_state and st.session_state['initial_prediction_run']:
    input_df_full = st.session_state['initial_input_df_full']
    initial_predicted_level_label = st.session_state['initial_prediction_label']
    selected_region_name = st.session_state['initial_region_name']

    st.subheader("Initial Input Data")
    st.dataframe(input_df_full[selected_features_names])

    if initial_predicted_level_label:
        st.subheader(f"Initial Predicted Nutrient Gap Level in {selected_region_name}: {initial_predicted_level_label}")

        initial_prediction_data = {'Nutrient Gap Level': [initial_predicted_level_label], 'Value': [1]}
        initial_prediction_df = pd.DataFrame(initial_prediction_data)

        # Use columns for side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(6, 4))
            sns.barplot(data=initial_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
            plt.title("Initial Prediction")
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

# --- Intervention Simulation ---
st.header("Intervention Simulation")
st.write("Select an intervention scenario and its intensity to see its potential impact on the nutrient gap level.")

# Updated intervention scenarios with features to be affected (not hardcoded multipliers)
intervention_scenarios = {
    "No Intervention": [],
    "Increase in Nutritious Food": [
        'avg_ca(mg)', 'avg_thiamin(mg)', 'avg_vitb12(mcg)', 'avg_vita(mcg)',
        'avg_riboflavin(mg)', 'avg_niacin(mg)', 'millet(mt)', 'sorghum(mt)'
    ],
    "Promote Fortified Foods": [
        'avg_ca(mg)', 'avg_thiamin(mg)', 'avg_vitb12(mcg)', 'avg_vita(mcg)',
        'avg_riboflavin(mg)', 'avg_niacin(mg)'
    ],
    "Improve Climate Resilience": [
        'pcfci', 'sorghum(mt)', 'millet(mt)','avg_vitb12(mcg)', 'avg_vita(mcg)',
        'avg_riboflavin(mg)', 'avg_niacin(mg)'
    ]
}

selected_scenario = st.selectbox("Select an intervention scenario:", list(intervention_scenarios.keys()))
# Add a slider for the user to select the percentage
intervention_percentage = st.slider(
    "Select the percentage of intervention impact (10% to 100%)",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

if st.button("Simulate Intervention"):
    if 'initial_input_df_full' in st.session_state:
        try:
            simulated_df_full = st.session_state['initial_input_df_full'].copy()
            scenario_features = intervention_scenarios[selected_scenario]

            # Calculate the dynamic multiplier based on the slider value
            percentage_decimal = intervention_percentage / 100.0

            # Apply the multiplier to the relevant features
            for feature in scenario_features:
                if feature in simulated_df_full.columns:
                    # 'pcfci' is a price index, so a positive intervention means a decrease in value
                    if feature == 'pcfci':
                        multiplier = 1 + (percentage_decimal * 5) # Corrected multiplier for pcfci
                    # All other features are nutrients or crop yields, so a positive intervention means an increase
                    else:
                        multiplier = 1 + (percentage_decimal * 5)
                    simulated_df_full[feature] *= multiplier
                else:
                    st.write(f"Warning: Scenario impacts feature '{feature}' not found in the simulation dataframe.")

            predicted_level_label = get_prediction_result(simulated_df_full, num_pipeline, model, nutrient_gap_labels)

            st.subheader(f"Simulation Result for {selected_scenario} at {intervention_percentage}% Impact")

            simulated_prediction_data = {'Nutrient Gap Level': [predicted_level_label], 'Value': [1]}
            simulated_prediction_df = pd.DataFrame(simulated_prediction_data)
            
            # Use the second column for the simulated graph
            with col2:
                plt.figure(figsize=(6, 4))
                sns.barplot(data=simulated_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
                plt.title("Simulated Prediction")
                plt.ylabel("")
                plt.yticks([])
                st.pyplot(plt)
                plt.clf()


            if predicted_level_label == 'Small Nutrient Gap':
                st.success(f"Based on the '{selected_scenario}' intervention with a {intervention_percentage}% impact, the predicted nutrient gap is small.")
            elif predicted_level_label == 'Significant Nutrient Gap':
                st.warning(f"Based on the '{selected_scenario}' intervention with a {intervention_percentage}% impact, the predicted nutrient gap is significant. This area may require targeted interventions.")
            else:
                st.error(f"Based on the '{selected_scenario}' intervention with a {intervention_percentage}% impact, the predicted nutrient gap is severe. Urgent intervention may be needed in this area.")

        except Exception as e:
            st.error(f"Error during simulation: {e}")
    else:
        st.warning("Please get the initial prediction first before simulating interventions.")

