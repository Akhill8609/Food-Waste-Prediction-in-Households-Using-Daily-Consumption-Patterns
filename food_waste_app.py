# ===============================
# Complete Food Waste Prediction Project (Streamlit App)
# ===============================

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv('food_waste_dataset.csv')
st.markdown("""
<div style='text-align: center;'>
    <h1>Food Waste Prediction in Households Using Daily Consumption Patterns</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Show dataset preview (scrollable)
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df, width=1000, height=400)  # Scrollable window

# -----------------------------
# Average Food Waste Charts
# -----------------------------
st.subheader("Average Food Waste by Meal Type")
avg_by_meal = df.groupby('Meal_Type')['Food_Waste_kg_per_Week'].mean()
st.bar_chart(avg_by_meal)

st.subheader("Average Food Waste by Household Size")
avg_by_household = df.groupby('Household_Size')['Food_Waste_kg_per_Week'].mean()
st.line_chart(avg_by_household)

# -----------------------------
# User Input for Prediction
# -----------------------------
st.subheader("Predict Food Waste for Your Household")

household_size = st.number_input("Household Size (1-6)", min_value=1, max_value=6, value=3)
daily_meals = st.number_input("Daily Meal Count (1-3)", min_value=1, max_value=3, value=2)
meal_type = st.selectbox("Meal Type", df['Meal_Type'].unique())
shopping_habit = st.selectbox("Shopping Habit", df['Shopping_Habit'].unique())
storage = st.selectbox("Storage Availability", df['Storage_Availability'].unique())
awareness = st.selectbox("Awareness of Waste Management", df['Awareness_of_Waste_Management'].unique())
leftovers = st.selectbox("Leftovers Frequency", df['Leftovers_Frequency'].unique())
income = st.selectbox("Income Range", df['Income_Range'].unique())
cooking_pref = st.selectbox("Cooking Preference", df['Cooking_Preference'].unique())

# -----------------------------
# Load Model, Scaler, and Label Encoders
# -----------------------------
try:
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    categorical_cols = ['Meal_Type','Shopping_Habit','Storage_Availability',
                        'Awareness_of_Waste_Management','Leftovers_Frequency',
                        'Income_Range','Cooking_Preference']

except FileNotFoundError:
    st.error("Model files not found! Please run the training script first.")
    st.stop()

# -----------------------------
# Prepare input for prediction
# -----------------------------
input_dict = {
    'Household_Size': [household_size],
    'Daily_Meal_Count': [daily_meals],
    'Meal_Type': [meal_type],
    'Shopping_Habit': [shopping_habit],
    'Storage_Availability': [storage],
    'Awareness_of_Waste_Management': [awareness],
    'Leftovers_Frequency': [leftovers],
    'Income_Range': [income],
    'Cooking_Preference': [cooking_pref]
}

input_df = pd.DataFrame(input_dict)

# Encode categorical columns
for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
numeric_cols = ['Household_Size', 'Daily_Meal_Count']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    try:
        predicted_waste = rf_model.predict(input_df)[0]
        st.subheader("Predicted Weekly Food Waste")
        st.write(f"Your predicted food waste is: **{predicted_waste:.2f} kg/week**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# -----------------------------
# Outro Section
# -----------------------------
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 15px; margin-top: 40px;'>
    <h3 style='color: #4CAF50; font-family: "Courier New", Courier, monospace;'>
        Thank you for using this app!<br>
        <span style='font-size: 18px;'>Developed with ❤️ by <strong>Akhil</strong></span>
    </h3>
</div>
""", unsafe_allow_html=True)


# ===============================
# Step 9: Interpretation of Results & Saving Files
# ===============================
# Feature Importance (Random Forest):
# - Leftovers_Frequency → highest impact
# - Awareness_of_Waste_Management → second highest
# - Household_Size → significant
#
# Model Comparison:
# - Random Forest performed best (R², MAE, RMSE from output)
# - Linear Regression and Decision Tree are slightly worse
#
# Observations:
# - Larger households produce more food waste
# - Bulk buyers with low awareness generate more waste
# - Batch cooking slightly increases waste in some scenarios
#
# Implications:
# - Educating households on leftovers and storage can reduce waste
# - Awareness campaigns and better shopping habits could help

# Save Random Forest model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# Save the LabelEncoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


print("Model, scaler, and label encoders saved successfully!")

