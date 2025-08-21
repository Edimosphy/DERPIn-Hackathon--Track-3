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
    st.error(f"An unexpected error occurred during setup: {e}")
    st.stop()

# --- Preprocessing Pipeline ---
num_pipeline = Pipeline([
    ("scalar", StandardScaler()),
    ("variance", VarianceThreshold(threshold=0.1)),
    ("selector", SelectKBest(score_func=f_classif))
])

X_for_pipeline_fit = nutrient_gap_encoded.drop(
    columns=["nutrient_gap_level_encoded", "category", "mnari", "nutrient_gap_level"],
    errors='ignore'
)
y_for_pipeline_fit = nutrient_gap_encoded["nutrient_gap_level_encoded"]

num_pipeline.fit(X_for_pipeline_fit, y_for_pipeline_fit)
selected_features_names = X_for_pipeline_fit.columns[num_pipeline.named_steps["selector"].get_support()]

# --- Utility Function for Prediction ---
def get_prediction_result(input_df_full, pipeline, model, labels):
    try:
        processed_data = pipeline.transform(input_df_full)
        predicted_level_encoded = model.predict(processed_data)[0]
        predicted_level_label = labels.get(int(predicted_level_encoded), 'Unknown')
        return predicted_level_label
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- Streamlit UI ---
st.title("Nutrient Gap Prediction & Intervention Simulation")

st.header("User Input Guide")
st.write("Input the corresponding numerical codes for your inputs")

# Regional codes
st.write("### Regional Codes Guide")
st.write("""
| Code | Region Name  |
|:----:|:-------------|
| 0    | Ashanti      |
| 1    | Brong Ahafo  |
| 2    | Central      |
| 3    | Eastern      |
| 4    | Greater Accra|
| 5    | Northern     |
| 6    | Upper East   |
| 7    | Upper West   |
| 8    | Volta        |
| 9    | Western      |
""")

# PCFCI codes
st.write("### PCFCI Codes Guide")
st.write("""
| Value | Vulnerability Level |
|:-----:|:--------------------|
| 0.0   | Vulnerable          |
| 1.0   | Not Vulnerable      |
""")

# VCCI codes
st.write("### VCCI Codes Guide")
st.write("""
| Value | Vulnerability Level |
|:-----:|:--------------------|
| 0.0   | Not Vulnerable      |
| 1.0   | Vulnerable          |
""")

category_region_map = nutrient_gap_original[['category', 'region']].drop_duplicates().set_index('category')['region'].to_dict()

feature_labels = {
    'category': 'Region (Category)',
    'pcfci': 'Per Capita Food Consumption Index (PCFCI)',
    'avg_kcalories': 'Average Kilocalories (%)',
    'avg_ca(mg)': 'Average Calcium (mg) (%)',
    'avg_folate(mcg)': 'Average Folate (mcg) (%)',
    'avg_iron(mg)': 'Average Iron (mg) (%)',
    'avg_niacin(mg)': 'Average Niacin (mg) (%)',
    'avg_riboflavin(mg)': 'Average Riboflavin (mg) (%)',
    'avg_thiamin(mg)': 'Average Thiamin (mg) (%)',
    'avg_vita(mcg)': 'Average Vitamin A (mcg) (%)',
    'avg_vitb12(mcg)': 'Average Vitamin B12 (mcg) (%)',
    'avg_vitb6(mg)': 'Average Vitamin B6 (mg) (%)',
    'avg_zinc(mg)': 'Average Zinc (mg) (%)',
    'vcci': 'Vulnerability to Climate Change Index (VCCI)',
    'maize(mt)': 'Maize Production (amt)',
    'rice(mt)': 'Rice Production (amt)',
    'sorghum(mt)': 'Sorghum Production (amt)',
    'cassava(mt)': 'Cassava Production (amt)',
    'millet(mt)': 'Millet Production (amt)',
    'price': 'Commodity Price',
}
for col in X_for_pipeline_fit.columns:
    if col.startswith('commodity_'):
        feature_labels[col] = f"Commodity: {col.replace('commodity_', '')} (0 or 1)"

st.header("Nutrient Gap Prediction")
initial_input_data = {}

for feature in selected_features_names:
    display_label = feature_labels.get(feature, feature)
    if feature == 'category':
        category_options = sorted(list(category_region_map.keys()))
        selected_category = st.selectbox(display_label, options=category_options,
                                         format_func=lambda x: category_region_map.get(x, x))
        initial_input_data[feature] = selected_category
    elif feature in X_for_pipeline_fit.columns:
        min_val = float(X_for_pipeline_fit[feature].min())
        max_val = float(X_for_pipeline_fit[feature].max())
        mean_val = float(X_for_pipeline_fit[feature].mean())
        step_val = 1.0 if X_for_pipeline_fit[feature].nunique() <= 2 else 0.1
        initial_input_data[feature] = st.number_input(display_label, min_value=min_val,
                                                      max_value=max_val, value=mean_val,
                                                      step=step_val)
    else:
        st.write(f"Warning: Input for {feature} not directly available. Using default 0.")
        initial_input_data[feature] = 0.0

if st.button("Get Initial Prediction"):
    try:
        input_df_full = pd.DataFrame(np.zeros((1, X_for_pipeline_fit.shape[1])), columns=X_for_pipeline_fit.columns)
        for feature, value in initial_input_data.items():
            if feature in input_df_full.columns:
                input_df_full[feature] = value

        initial_predicted_level_label = get_prediction_result(input_df_full, num_pipeline, model, nutrient_gap_labels)
        
        if initial_predicted_level_label:
            st.subheader(f"Initial Predicted Nutrient Gap Level: {initial_predicted_level_label}")

            initial_prediction_df = pd.DataFrame({'Nutrient Gap Level': [initial_predicted_level_label], 'Value': [1]})
            plt.figure(figsize=(6, 4))
            sns.barplot(data=initial_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
            plt.title("Initial Predicted Nutrient Gap Level")
            plt.ylabel("")
            plt.yticks([])
            st.pyplot(plt)
            plt.close()

            if initial_predicted_level_label == 'Small Nutrient Gap':
                st.success("The prediction indicates a small nutrient gap. Continue monitoring and promoting healthy diets.")
            elif initial_predicted_level_label == 'Significant Nutrient Gap':
                st.warning("The prediction indicates a significant nutrient gap. Consider implementing targeted interventions.")
            else:
                st.error("The prediction indicates a severe nutrient gap. Urgent interventions are highly recommended.")

            st.session_state['initial_input_df_full'] = input_df_full
    except Exception as e:
        st.error(f"Error during initial prediction: {e}")

st.header("Intervention Simulation")
intervention_scenarios = {
    "No Intervention": {},
    "Increase in Nutritious Food": {
        'avg_ca(mg)': 1.1, 'avg_thiamin(mg)': 1.1, 'avg_vitb12(mcg)': 1.1,
        'pcfci': 0.9, 'avg_vita(mcg)': 1.1, 'avg_riboflavin(mg)': 1.1,
        'avg_niacin(mg)': 1.1, 'millet(mt)': 1.2, 'sorghum(mt)': 1.2
    },
    "Promote Fortified Foods": {
        'avg_ca(mg)': 1.1, 'avg_thiamin(mg)': 1.1, 'avg_vitb12(mcg)': 1.1,
        'pcfci': 0.9, 'avg_vita(mcg)': 1.1, 'avg_riboflavin(mg)': 1.1,
        'avg_niacin(mg)': 1.1
    },
    "Improve Climate Resilience": {
        'pcfci': 0.9, 'sorghum(mt)': 1.3, 'millet(mt)': 1.3
    }
}

selected_scenario = st.selectbox("Select an intervention scenario:", list(intervention_scenarios.keys()))

if st.button("Simulate Intervention"):
    if 'initial_input_df_full' in st.session_state:
        try:
            simulated_df_full = st.session_state['initial_input_df_full'].copy()
            for feature, multiplier in intervention_scenarios[selected_scenario].items():
                if feature in simulated_df_full.columns:
                    simulated_df_full[feature] *= multiplier

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
