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