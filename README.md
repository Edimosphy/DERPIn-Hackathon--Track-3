# DERPIn Hackathon Track 3 Documentation

# Inclusive Nutrition Policies: Nutrient & Hidden Hunger Insights (Case study = Ghana)
This project is developed as part of the Hackathon Challenge Track 3, focuses on identifying nutrient gaps and hidden hunger in Africa using data-driven approaches. The goal is to provide insights and tools for policymakers and community leaders to develop inclusive and targeted nutrition policies.

## Project Objective
The main objectives of this project are:
- Build AI-driven tools to predict nutrition gaps.
- Display predictions in mobile apps or interactive dashboards to guide policy and community action.

## Data Sources
The analysis leverages data from multiple sources for the year 2022, including:
- Ghana Food Security and Nutrition Indicators (FAOSTAT, HDX)- Other reliable source
  - dataset name: ghana_food_price_FAO (final dataset used)
  - Link: <https://data.humdata.org/dataset/faostat-food-security-indicators-for-ghana> (raw data)
- AGWAA API Documentation - Ghana (2022)
  - dataset name: aagwa_data_prod (final dataset used)
  - Link: <https://www.aagwa.org/Ghana/data?p=Ghana> - Under Publication (raw data)
- Food System Crisis Observatory and Response (FS-COR) Platform
  - dataset name: Ghana_nutrient_dataset (final dataset used)
  - Link: <https://fs-cor.org/Ghana/> (raw data)
    
## Methodology
The project followed these key steps:
- Study of the AGWAA and FS-COR dataset to gain more insights for the prediction model building.
  - After due studies, the following indicators was identified as the nutrient adequency indicators to identify the nutrient gap.
  - For the AGWAA dataset: These dataset was majorly about the the food production and its yield. The dataset was obtained by merging the dataset from the publication for food such as cassava, rice, sorghum, millet.These dataset was obtained from year 2020 because of its consistent.
  - For the FS-COR, the focus was on Mean Nutrient Adequacy Ratio Index across the region as this was seen to affected the nutrient consumption ratio. Further exploration was done and it was observed that region tends to consume a particular meal even though it contain less nutrition but was affordable. This exploration leds to using other indicator such as Vulnerability to Climate change (to checkmate if it affected AGWAA dataset which has crop production) and Per capita Food consumption index (to check the aamount of food per on price and quality).
  - With this due observation, the Ghana Food Security and Nutrition Indicators dataset which contain the price of commodity per region was also used to check the impact of price in nutrition adequacy.
- Data Collection and Integration: Gathering and merging of the dataset from the specified sources into jupyter notebook for proper analysis.
- Data Preprocessing: Cleaning, handling missing values and outliers, and preparing the data for analysis and modeling. This included one-hot encoding of categorical features and creating a classified target variable for nutrient gap levels.
- Exploratory Data Analysis (EDA): Analyzing data distributions, relationships between variables, and identifying patterns related to nutrient gaps through visualizations and statistical summaries.
- Feature Engineering and Selection: Applying techniques to transform and select the most relevant features for the predictive model.
- Model Development: Training and evaluating classification models to predict nutrient gap levels. A Random Forest Classifier was selected for the final application.
- Intervention Simulation: Implementing a feature to simulate the potential impact of different interventions on nutrient gap levels.
- Visualization and Reporting: Developing an interactive Streamlit dashboard to visualize the data, provide predictions, and simulate interventions.

## Model Development 
This prediction was based using a classification model algorithms.
The model training and evaluation was done using the following models:
- Logistics Regression
- Decision Tree Classifier
- Random Forest Classifier.

These models had a 100% accuracy score and 100% F1-score showing its ability to recall and precision.

The models were checked to confirm there was no data leakage, overfitting.
- This high performance might be as the results of the highly correlation features with the target variables.
- The random forest model was saved because of its characteristics  and to capture more non-linear relationship with future real dataset.

## Feature Importance of Model
The feature importance of prediction model was checked using the random forest classifier
![IMG_20250822_152422_189](https://github.com/user-attachments/assets/777aa660-e67a-418c-9ae3-bd5f6575cfcc)

The features importance were used as the basis for user input on the prediction model.



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

## Usage 
Usage of the application is broken down into two main interfaces, accessible via the sidebar on the home page using streamlit web app via <https://derpin-nutrient-gap.streamlit.app>
- Data Visualizations: This section presents interactive charts and graphs that allow you to explore key findings, such as regional disparities in nutrient adequacy and the impact of different commodities on nutritional status.
- Prediction & Simulation: This interface allows you to input specific data points to get a prediction on the nutrient gap level. It also includes a simulation feature where you can model the effects of different interventions and see how they would change the predicted outcome.

##  Result
### Home page of the app
<img width="800" height="1280" alt="Screenshot_20250822-024200" src="https://github.com/user-attachments/assets/ca33dfa6-ef5f-44b7-bc43-10596e217f69" />

### Result After Prediction
<img width="800" height="1280" alt="Screenshot_20250822-024523" src="https://github.com/user-attachments/assets/afeb343c-9771-42d3-9288-cb23324ebc84" />

### Result After Applying Intervention
<img width="800" height="1280" alt="Screenshot_20250822-024644" src="https://github.com/user-attachments/assets/37d149d2-40e8-4ec6-90e0-018d771959b7" />


## Challenges
- The dataset from AGWAA API Documentation and Food System Crisis Observatory and Response (FS-COR) Platform were hard to collect as the Api of the file was not accessible.
- This was solved by manually collected dataset from AGWAA API publication and sorting of the FS-COR into csv file.

## Link to Final dataset used
- Ghana_nutrient_dataset:<https://drive.google.com/file/d/1-fe2BXQHXdscddCzNL3aW27pVfD9MwjP/view?usp=drivesdk>
- aagwa_data_prod: <https://drive.google.com/file/d/102Crelz3wUxvK9HdMXDLm8ZvqcEXOJ7u/view?usp=drivesdk>
- ghana_food_price_FA0: [Uploading wfp_food_prices_gha (1).csv…]()

  
