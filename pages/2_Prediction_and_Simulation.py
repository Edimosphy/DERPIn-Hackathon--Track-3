import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline


try:
    # Load data and model
    nutrient_gap_original = pd.read_csv('nutrient_gap.csv')
    nutrient_gap_encoded = pd.read_csv('nutrient_gap_encoded.csv')
    if 'nutrient_gap_level_encoded' in nutrient_gap_encoded.columns:
         nutrient_gap_encoded['nutrient_gap_level_encoded'] = nutrient_gap_encoded['nutrient_gap_level_encoded'].astype(int)
    model = joblib.load('nutrient_gap_model.pkl')

    st.title("Nutrient Gap Prediction & Intervention Simulation")

    # Recreate the preprocessing pipeline and fit it
    num_pipeline = Pipeline([
        ("scalar", StandardScaler()),
        ("variance", VarianceThreshold(threshold=0.1)),
        ("selector", SelectKBest(score_func=f_classif))
    ])

    X_for_pipeline_fit = nutrient_gap_encoded.drop(columns=["nutrient_gap_level_encoded", "category", "mnari", "nutrient_gap_level"], errors='ignore')
    y_for_pipeline_fit = nutrient_gap_encoded["nutrient_gap_level_encoded"]
    num_pipeline.fit(X_for_pipeline_fit, y_for_pipeline_fit)

    selected_features_names = X_for_pipeline_fit.columns[num_pipeline.named_steps["selector"].get_support()]
    category_region_map = nutrient_gap_original[['category', 'region']].drop_duplicates().set_index('category')['region'].to_dict()

    feature_labels = {
        'category': 'Region (Category)',
        'pcfci': 'Per Capita Food Consumption Index',
        'avg_kcalories': 'Average Kilocalories',
        'avg_ca(mg)': 'Average Calcium (mg)',
        'avg_folate(mcg)': 'Average Folate (mcg)',
        'avg_iron(mg)': 'Average Iron (mg)',
        'avg_niacin(mg)': 'Average Niacin (mg)',
        'avg_riboflavin(mg)': 'Average Riboflavin (mg)',
        'avg_thiamin(mg)': 'Average Thiamin (mg)',
        'avg_vita(mcg)': 'Average Vitamin A (mcg)',
        'avg_vitb12(mcg)': 'Average Vitamin B12 (mcg)',
        'avg_vitb6(mg)': 'Average Vitamin B6 (mg)',
        'avg_zinc(mg)': 'Average Zinc (mg)',
        'vcci': 'Vulnerability to Climate Change Index',
        'maize(mt)': 'Maize Production (mt)',
        'rice(mt)': 'Rice Production (mt)',
        'sorghum(mt)': 'Sorghum Production (mt)',
        'cassava(mt)': 'Cassava Production (mt)',
        'millet(mt)': 'Millet Production (mt)',
        'price': 'Commodity Price',
    }
    
    # Add one-hot encoded labels
    for col in X_for_pipeline_fit.columns:
        if col.startswith('commodity_'):
            feature_labels[col] = f"Commodity: {col.replace('commodity_', '')} (0 or 1)"

    st.header("Nutrient Gap Prediction")
    st.write("Enter the feature values below to get the initial nutrient gap prediction.")

    initial_input_data = {}
    for feature in selected_features_names:
        display_label = feature_labels.get(feature, feature)
        if feature == 'category':
            category_options = sorted(list(category_region_map.keys()))
            selected_category = st.selectbox(display_label, options=category_options, format_func=lambda x: category_region_map.get(x, x))
            initial_input_data[feature] = selected_category
        elif feature in X_for_pipeline_fit.columns:
            min_val = float(X_for_pipeline_fit[feature].min())
            max_val = float(X_for_pipeline_fit[feature].max())
            mean_val = float(X_for_pipeline_fit[feature].mean())
            step_val = 1.0 if X_for_pipeline_fit[feature].nunique() <= 2 else None
            initial_input_data[feature] = st.number_input(display_label, min_value=min_val, max_value=max_val, value=mean_val, step=step_val)
        else:
             st.write(f"Warning: Input for {feature} not directly available. Using default 0.")
             initial_input_data[feature] = 0.0


    if st.button("Get Initial Prediction"):
        try:
            input_df_full = pd.DataFrame(np.zeros((1, X_for_pipeline_fit.shape[1])), columns=X_for_pipeline_fit.columns)
            for feature, value in initial_input_data.items():
                 if feature in input_df_full.columns:
                      input_df_full[feature] = value
                 else:
                      st.write(f"Warning: Input feature '{feature}' not found in the pipeline's expected columns.")

            initial_processed = num_pipeline.transform(input_df_full)
            initial_predicted_level_encoded = model.predict(initial_processed)[0]
            nutrient_gap_labels = {0: 'Small Nutrient Gap', 1: 'Significant Nutrient Gap', 2: 'Severe Nutrient Gap'}
            initial_predicted_level_label = nutrient_gap_labels.get(initial_predicted_level_encoded, 'Unknown')
            st.subheader(f"Initial Predicted Nutrient Gap Level: {initial_predicted_level_label}")

            initial_prediction_data = {'Nutrient Gap Level': [initial_predicted_level_label], 'Value': [1]}
            initial_prediction_df = pd.DataFrame(initial_prediction_data)
            plt.figure(figsize=(6, 4))
            sns.barplot(data=initial_prediction_df, x='Nutrient Gap Level', y='Value', palette='viridis')
            plt.title("Initial Predicted Nutrient Gap Level")
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
        "Increase Protein Consumption": {
            'avg_iron(mg)': 1.1,
            'avg_zinc(mg)': 1.1,
            'avg_vitb12(mcg)': 1.1
        },
        "Promote Fortified Foods": {
            'avg_vita(mcg)': 1.2,
            'avg_folate(mcg)': 1.2
        },
        "Improve Climate Resilience": {
            'vcci': 0.9,
            'maize(mt)': 1.1,
            'rice(mt)': 1.1
        }
    }

    selected_scenario = st.selectbox("Select an intervention scenario:", list(intervention_scenarios.keys()))

    if st.button("Simulate Intervention"):
        if 'initial_input_df_full' in st.session_state:
            try:
                simulated_df_full = st.session_state['initial_input_df_full'].copy()
                scenario_impact = intervention_scenarios[selected_scenario]

                for feature, multiplier in scenario_impact.items():
                     if feature in simulated_df_full.columns:
                          simulated_df_full[feature] *= multiplier
                     else:
                          st.write(f"Warning: Scenario impacts feature '{feature}' not found in the simulation dataframe.")

                simulation_processed = num_pipeline.transform(simulated_df_full)
                predicted_level_encoded = model.predict(simulation_processed)[0]
                predicted_level_label = nutrient_gap_labels.get(predicted_level_encoded, 'Unknown')
                st.subheader(f"Predicted Nutrient Gap Level after {selected_scenario}: {predicted_level_label}")

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


    st.header("Recommendations")
    st.write("""
    Based on the analysis, the following recommendations can be considered to address nutrient gaps in Ghana:

    - **Investigate the impact of climate change:** Conduct further research into how climate change specifically affects agricultural production in vulnerable regions like Northern Ghana. Implement climate-resilient agricultural practices.
    - **Regulate commodity prices:** Implement policies to regulate the prices of highly nutritious commodities to make them more affordable and accessible to vulnerable populations.
    - **Promote diversified agriculture:** Encourage the production of a wider variety of nutritious foods beyond staple crops, including protein-rich sources like poultry and fish, and nutrient-dense crops like sorghum and millet.
    - **Promote nutritional education:** Implement educational programs in schools and communities to raise awareness about the importance of balanced diets and proper nutrition for healthy living and productivity.
    - **Support food fortification:** Encourage and support initiatives for food fortification, such as adding Vitamin A to cassava, to improve the nutritional content of commonly consumed foods.
    """)

except FileNotFoundError:
    st.error("Error: Required data files ('nutrient_gap.csv', 'nutrient_gap_encoded.csv', or 'nutrient_gap_model.pkl') were not found. Please ensure they are in the same directory as the app.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")