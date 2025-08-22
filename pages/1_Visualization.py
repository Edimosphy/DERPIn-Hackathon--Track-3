import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

# --- Global Variables and Data Loading ---
try:
    # Load the original data for visualizations
    nutrient_gap_original = pd.read_csv('nutrient_gap.csv')
    
    # Load the encoded data for the correlation matrix
    nutrient_gap_encoded = pd.read_csv('nutrient_gap_encoded.csv')

except FileNotFoundError as e:
    st.error(f"Error: Required data files not found. {e}. Please ensure the files are in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during setup: {e}")
    st.stop()


# --- Data Preprocessing for Top/Less 5 Commodities ---

# Calculate the mean MNARI for each commodity
commodity_mnari_mean = nutrient_gap_original.groupby('commodity')['mnari'].mean().reset_index()

# Sort to get the top 5 (lowest MNARI, most significant gap) and less 5 (highest MNARI, least significant gap)
# A lower MNARI means a more severe gap, so we sort in ascending order.
top_5_commodities = commodity_mnari_mean.sort_values(by='mnari', ascending=True).head(5)
less_5_commodities = commodity_mnari_mean.sort_values(by='mnari', ascending=False).head(5)

# Filter the original dataframes to create the specific data for the plots
top_5_nutrient_gap = nutrient_gap_original[
    nutrient_gap_original['commodity'].isin(top_5_commodities['commodity'])
].copy()

less_5_nutrient_gap = nutrient_gap_original[
    nutrient_gap_original['commodity'].isin(less_5_commodities['commodity'])
].copy()


# --- Streamlit UI and Plots ---

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

st.write("This chart shows the average Mean Nutrient Adequacy Ratio Index (MNARI) for each region in Ghana. Higher MNARI values indicate a higher nutrient gap.")


st.header("Mean Nutrient Adequacy Ratio Index by Commodity")
# Mean Nutrient Adequacy Ratio Index by Commodity
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

st.write("This chart displays the average MNARI for different commodities. Commodities with higher MNARI might be less accessible or consumed in areas with high nutrient gaps due to many factors such as price and low yields.")


st.header("Price by Commodity")
# Price Vs Commodity
plt.figure(figsize= (15,8))
ax = sns.barplot(data= nutrient_gap_original, x= "commodity", y= "price", errorbar=None)
plt.xlabel("Commodity", fontsize= 12, fontweight= "bold")
plt.ylabel("Price", fontsize= 12, fontweight= "bold")
plt.title("Price by Commodity", fontsize= 14, fontweight= "bold")
plt.xticks(rotation=90)
# Add text labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f')
st.pyplot(plt)
plt.clf()
st.write("This chart displays the price of different commodities. Commodities with higher price might be less accessible or consumed in areas thus leading to high nutrient gaps and hunger.")


st.header("Per Capita Food Consumption Index by Region")
# Region Vs PCFCI
plt.figure(figsize= (12,5))
ax = sns.barplot(data= nutrient_gap_original, x ="region", y ="pcfci", errorbar= None)
plt.title("Per Capita Food Consumption Index by Region", fontsize= 14, fontweight= "bold")
plt.xlabel("Region", fontsize= 12, fontweight= "bold")
plt.ylabel("Per Capita Food Consumption Index (PCFCI)", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)
#Add text labels on top of each bar
for container in ax.containers:
  ax.bar_label(container, fmt= "%.2f")
st.pyplot(plt)
plt.clf()
st.write("This chart displays the average amount of food available for consumption in a region. Lower PCFCI values indicates severe nutrient gaps in the region.")

st.header("Vulnerability to Climate Change Index by Region")
# Region Vs VCCI
plt.figure(figsize=(12, 5))
ax= sns.barplot(data= nutrient_gap_original, x= "region", y= "vcci", errorbar= None)
plt.title("Vulnerability to Climate Change Index by Region", fontsize= 14, fontweight= "bold")
plt.xlabel("Region", fontsize= 12, fontweight= "bold")
plt.ylabel("Vulnerability to Climate Change Index (VCCI)", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)
#Add text labels on top of each bar
for container in ax.containers:
  ax.bar_label(container, fmt= "%.2f")
st.pyplot(plt)
plt.clf()
st.write("This chart displays effect of climate change. Higher VCCI leads to low agricultural production and yield thus severe nutrient gap")

st.header("Per Capita Food Consumption Index by Commodity")
# Commodity Vs PCFCI
plt.figure(figsize= (15,8))
ax = sns.barplot(data= nutrient_gap_original, x= "commodity", y= "pcfci", errorbar=None)
plt.title("Per Capita Food Consumption Index by Commodity", fontsize= 14, fontweight= "bold")
plt.xlabel("Commodity", fontsize= 12, fontweight= "bold")
plt.ylabel("Per Capita Food Consumption Index (PCFCI)", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)
# Add text labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
st.pyplot(plt)
plt.clf()
st.write("This chart displays the average amount of a particular food available for consumption. Lower PCFCI values indicates severe nutrient deficiency.")


st.header("Vulnerability to Climate Change Index by Commodity")
# Commodity Vs VCCI
plt.figure(figsize= (15,8))
ax = sns.barplot(data= nutrient_gap_original, x= "commodity", y= "vcci", errorbar=None)
plt.title("Vulnerability to Climate Change Index by Commodity", fontsize= 14, fontweight= "bold")
plt.xlabel("Commodity", fontsize= 12, fontweight= "bold")
plt.ylabel("Vulnerability to Climate Change Index (VCCI)", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)
# Add text labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
st.pyplot(plt)
plt.clf()
st.write("This chart displays the effect of climate change across agricultural production. Higher VCCI leads to low agricultural production and yield thus severe nutrient gap")

st.header("Top 5 Commodities that is Highly Consumable Per Region")
# Top 5 Commodity Vs Price Vs Region
plt.figure(figsize= (15, 10))
ax = sns.barplot(data=top_5_nutrient_gap, x="region", y="price", hue="commodity", errorbar=None)
plt.title("Price of Top 5 Commodities by Region", fontsize= 14, fontweight= "bold")
plt.xlabel("Region", fontsize= 12, fontweight= "bold")
plt.ylabel("Price", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)

# Label the box
for container in ax.containers:
  ax.bar_label(container, fmt='%.1f')

st.pyplot(plt)
plt.clf()

st.write("This chart illustrates the prices of the top 5 most consume commodities across different regions. This helps to visualize regional price variations for key staples foods.")


st.header("5 Commodities that is Less Consumable Per Region")
# Less 5 Commodity Vs Price Vs Region
plt.figure(figsize= (15, 10))
ax = sns.barplot(data=less_5_nutrient_gap, x="region", y="price", hue="commodity", errorbar=None)
plt.title("Price of 5 Least Significant Nutrient-Gap Commodities by Region", fontsize= 14, fontweight= "bold")
plt.xlabel("Region", fontsize= 12, fontweight= "bold")
plt.ylabel("Price", fontsize= 12, fontweight= "bold")
plt.xticks(rotation=90)

# Label the box
for container in ax.containers:
  ax.bar_label(container, fmt='%.1f')

st.pyplot(plt)
plt.clf()

st.write("This chart shows the prices of the 5 less consumable commodities by region, which can provide insights into the cost of food sources.")


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
  - **Nutrient Gap Distribution:** The Mean Nutrient Adequacy Ratio Index (MNARI), used as the target variable, shows regional variations in nutrient adequacy. Some regions, particularly Eastern and Central, exhibit low MNARI values (lower nutrient gaps), while Upper West and Upper East show low MNARI values (severe nutrient gaps).
  - **Commodity Consumption Patterns:** Staple crops like Maize, Yam, and Plantains are the most consumed commodities, while highly nutritious options such as Rice (Paddy), Meat (Chicken, Local), and Cowpeas have lower consumption rates due to price rate.
  - **Price and Nutrient Gaps:** Commodities with higher nutritional value tend to have higher prices, potentially limiting access for vulnerable populations and contributing to nutrient gaps.
  - **Climate Change Vulnerability:** Regions with higher vulnerability to climate change, such as Northern and Upper West, appear to be associated with certain nutrient gap conditions. Climate conditions may impact agricultural production and food availability thus leading to hunger.

""")

st.header("Recommendations")
st.write("""
    Based on the analysis, the following recommendations can be considered to address nutrient gaps in Ghana:

    - **Investigate the impact of climate change:** Conduct further research into how climate change specifically affects agricultural production in vulnerable regions like Northern part of Ghana. Implement climate-resilient agricultural practices.
    - **Regulate commodity prices:** Implement policies to regulate the prices of highly nutritious commodities to make them more affordable and accessible to vulnerable populations.
    - **Promote diversified agriculture:** Encourage the production of a wider variety of nutritious foods beyond staple crops, including protein-rich sources like poultry and fish, and nutrient-dense crops like sorghum and millet.
    - **Promote nutritional education:** Implement educational programs in schools and communities to raise awareness about the importance of balanced diets and proper nutrition for healthy living and productivity.
    - **Support food fortification:** Encourage and support initiatives for food fortification, such as adding Vitamin A to cassava, to improve the nutritional content of commonly consumed foods.
    """)
