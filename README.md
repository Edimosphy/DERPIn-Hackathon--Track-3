# DERPIn-Hackathon--Track-3
# Inclusive Nutrition Policies: Ghana Nutrient & Hidden Hunger Insights
This project is developed as part of the Hackathon Challenge Track 3, focuses on identifying nutrient gaps and hidden hunger in Africa using data-driven approaches. The goal is to provide insights and tools for policymakers and community leaders to develop inclusive and targeted nutrition policies.

## Project Objective
The main objectives of this project are:
- Build AI-driven tools to predict nutrition gaps.
- Display predictions in mobile apps or interactive dashboards to guide policy and community action.

## Data Sources
The analysis leverages data from multiple sources for the year 2022, including:
- Ghana Food Security and Nutrition Indicators (FAOSTAT, HDX)
- AGWAA API Documentation - Ghana (2022)
- Food System Crisis Observatory and Response (FS-COR) Platform
  
## Methodology
The project followed these key steps:
- Data Collection and Integration: Gathering and merging data from the specified sources.
- Data Preprocessing: Cleaning, handling missing values and outliers, and preparing the data for analysis and modeling. This included one-hot encoding of categorical features and creating a classified target variable for nutrient gap levels.
- Exploratory Data Analysis (EDA): Analyzing data distributions, relationships between variables, and identifying patterns related to nutrient gaps through visualizations and statistical summaries.
- Feature Engineering and Selection: Applying techniques to transform and select the most relevant features for the predictive model.
- Model Development: Training and evaluating classification models to predict nutrient gap levels. A Random Forest Classifier was selected for the final application.
- Intervention Simulation: Implementing a feature to simulate the potential impact of different interventions on nutrient gap levels.
- Visualization and Reporting: Developing an interactive Streamlit dashboard to visualize the data, provide predictions, and simulate interventions.

## Key Insights
Based on the analysis of the Ghana nutrition data:
- Regional Disparities: There are significant regional differences in nutrient adequacy across Ghana, with Eastern and Central regions showing higher nutrient gaps compared to Upper West and Upper East.
- Dietary Patterns and Price: Consumption patterns are heavily skewed towards staple, carbohydrate-rich foods. More nutritious, protein-rich, and micronutrient-dense foods tend to be more expensive, making them less accessible to vulnerable populations and contributing to nutrient deficiencies.
- Climate Change Impact: Regions highly vulnerable to climate change, such as Northern Ghana, appear to experience conditions that negatively impact agricultural production and potentially exacerbate nutrient gaps.
- Micronutrient Deficiencies: Specific micronutrients like Vitamin B12, Calcium, and Niacin are identified as significant factors influencing nutrient gap levels, highlighting potential deficiencies in the diets of vulnerable populations.
- Model Effectiveness: The developed predictive model effectively identifies areas and populations vulnerable to nutrient gaps based on the available features.

## Recommendations:
- Targeted Interventions: Focus nutrition policies and interventions on regions identified with significant and severe nutrient gaps, such as Eastern and Central.
- Price Regulation and Accessibility of Nutrient Dense foods.
- Promote Diversified and Climate-Resilient Agriculture
- Enhance Nutritional Education
- Support Food Fortification on consumed staples food to address widespread micronutrient deficiencies.
- Further Research on Price Drivers: Investigate the underlying causes of price spikes in nutritious commodities, including the impact of climate change, market dynamics, and supply chain issues.
- Monitor and Evaluate: Continuously monitor nutrient gap levels and evaluate the impact of implemented interventions to adapt strategies as needed.

- Running the Application Locally
To run the Streamlit application on your local machine
