import streamlit as st

st.set_page_config(
    page_title="Inclusive Nutrition Policies- Nutrition & Hidden Hunger Insights",
    page_icon="🍲",
)

st.title("Welcome to the Inclusive Nutrient Gap App")
st.write("""
This application provides data visualizations, a predictive model, and intervention simulations to address nutrient gaps in Ghana.

👈 Use the sidebar to navigate between pages:
- **Data Visualizations**: Explore key data trends, key insights and recommendations.
- **Prediction Model and Simulation Intervention**: Use the predictive model to forecast nutrient gaps and simulate the impact of various interventions.
""")

st.header("Term for User Guide")
st.write("Know the meaning of term and what it stands for.")

# Display the information using markdown formatting
st.write("""
| Term |                Meaning of Term             | Vulnerability Level | Vunerability Level Code | Nutrient gap Indicator |
|:----:|:-------------------------------------------|:--------------------|:------------------------|:-----------------------|
| mnari|    Mean Nutrient Adequate Ratio Index      |     Vulnerable      |         1               |      Severe            |
|      |                                            |    Invulnerable     |         0               |    Not Severe          |
| pcfci|   Per Capita Food Consumption Index        |    Invulnerable     |         1               |    Not Severe          |
|      |                                            |     Vulnerable      |         0               |      Severe            |
| vcci |   Vulnerability to Climate Change Index    |     Vulnerable      |         1               |      Severe            |
|      |                                            |    Invulnerable     |         0               |    Not Severe          |
 
""")
