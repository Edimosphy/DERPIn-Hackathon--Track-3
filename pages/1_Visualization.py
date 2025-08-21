import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the original data for visualizations
try:
    nutrient_gap_original = pd.read_csv('nutrient_gap.csv')
except FileNotFoundError:
     st.error("Error: 'nutrient_gap.csv' not found. Please make sure the original data file is in the same directory as the app.")
     st.stop()
    
# Load the encoded data for the correlation matrix
try:
    nutrient_gap_encoded = pd.read_csv('nutrient_gap_encoded.csv')
except FileNotFoundError:
    st.error("Error: 'nutrient_gap_encoded.csv' not found. Please make sure the encoded data file is in the same directory as the app.")
    st.stop()



st.title("Data Visualizations")


st.header("Mean Nutrient Adequacy Ratio Index by Region")
# Mean Nutrient Adequacy Ratio Index by Region
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=nutrient_gap_original, x="region", y="mnari", errorbar=None)
plt.title("Mean Nutrient Adequacy Ratio Index by Region")
plt.xlabel("Region")
plt.ylabel("Mean Nutrient Adequacy Ratio Index (MNARI)")
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
st.pyplot(plt)
plt.clf()

st.write("This bar chart shows the average Mean Nutrient Adequacy Ratio Index (MNARI) for each region in Ghana. Lower MNARI values indicate a higher nutrient gap.")


# Mean Nutrient Adequacy Ratio Index by Commodity
st.header("Mean Nutrient Adequacy Ratio Index by Commodity")
plt.figure(figsize=(15, 8))
ax = sns.barplot(data=nutrient_gap_original, x="commodity", y="mnari", errorbar=None)
plt.title("Mean Nutrient Adequacy Ratio Index by Commodity")
plt.xlabel("Commodity")
plt.ylabel("Mean Nutrient Adequacy Ratio Index (MNARI)")
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
st.pyplot(plt)
plt.clf()

st.write("This bar chart displays the average MNARI for different commodities. Commodities with lower MNARI might be less accessible or consumed in areas with high nutrient gaps.")



st.header("Top 5 Commodity vs. Price vs. Region")
# Top 5 Commodity Vs Price Vs Region
plt.figure(figsize= (15,20))
ax = sns.barplot(data=top_5_nutrient_gap, x="region", y= "price", hue="commodity", errorbar=None)
plt.title("Price of High Consume Commodity by Region", fontsize= 14, fontweight= "bold")
plt.xlabel("Region", fontsize= 12, fontweight= "bold")
plt.ylabel("Price", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)

# Label the box
for container in ax.containers:
  ax.bar_label(container, fmt='%.1f')

st.pyplot(plt)
plt.clf()

st.write("This bar chart illustrates the prices of the top 5 most consumed commodities across different regions. This helps to visualize regional price variations for key staples.")


st.header("Less 5 Commodity vs. Price vs. Region")
# Less 5 Commodity Vs Price Vs Region
plt.figure(figsize= (15,20))
ax = sns.barplot(data=less_5_nutrient_gap, x="region", y="price", hue="commodity", errorbar=None)
plt.title("Price of 5 Less Consume Commodity by Region", fontsize= 14, fontweight= "bold")
plt.xlabel("Region", fontsize= 12, fontweight= "bold")
plt.ylabel("Price", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)

# Label the box
for container in ax.containers:
  ax.bar_label(container, fmt='%.1f')

st.pyplot(plt)
plt.clf()

st.write("This chart shows the prices of the 5 least consumed commodities by region, which can provide insights into the cost of less common food sources.")

# --- END OF NEW PLOTS ---


# Correlation Matrix
st.header("Correlation Matrix of Numerical Features")
numeric_cols_encoded = nutrient_gap_encoded.select_dtypes(include=np.number).drop(columns=['nutrient_gap_level_encoded'], errors='ignore')
corr_matrix = numeric_cols_encoded.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation Matrix')
st.pyplot(plt)
plt.clf()

st.write("The correlation matrix shows the relationships between the numerical features and the one-hot encoded commodity features in the dataset.")


st.header("Key Insight")
st.write("""
  - **Nutrient Gap Distribution:** The Mean Nutrient Adequacy Ratio Index (MNARI), used as the target variable, shows regional variations in nutrient adequacy. Some regions, particularly Eastern and Central, exhibit lower MNARI values (higher nutrient gaps), while Upper West and Upper East show higher MNARI values (lower nutrient gaps).
  - **Commodity Consumption Patterns:** Staple crops like Maize, Yam, and Plantains are the most consumed commodities, while highly nutritious options such as Rice (Paddy), Meat (Chicken, Local), and Cowpeas have lower consumption rates.
  - **Price and Nutrient Gaps:** Commodities with higher nutritional value tend to have higher prices, potentially limiting access for vulnerable populations and contributing to nutrient gaps.
  - **Climate Change Vulnerability:** Regions with higher vulnerability to climate change, such as Northern and Upper West, appear to be associated with certain nutrient gap conditions. Climate conditions may impact agricultural production and food availability.

""")

st.header("Recommendations")
st.write("""
    Based on the analysis, the following recommendations can be considered to address nutrient gaps in Ghana:

    - **Investigate the impact of climate change:** Conduct further research into how climate change specifically affects agricultural production in vulnerable regions like Northern Ghana. Implement climate-resilient agricultural practices.
    - **Regulate commodity prices:** Implement policies to regulate the prices of highly nutritious commodities to make them more affordable and accessible to vulnerable populations.
    - **Promote diversified agriculture:** Encourage the production of a wider variety of nutritious foods beyond staple crops, including protein-rich sources like poultry and fish, and nutrient-dense crops like sorghum and millet.
    - **Promote nutritional education:** Implement educational programs in schools and communities to raise awareness about the importance of balanced diets and proper nutrition for healthy living and productivity.
    - **Support food fortification:** Encourage and support initiatives for food fortification, such as adding Vitamin A to cassava, to improve the nutritional content of commonly consumed foods.
    """)
