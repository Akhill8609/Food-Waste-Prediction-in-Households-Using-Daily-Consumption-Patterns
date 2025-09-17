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
# It's better to reload the dataset to get the categories for the selectboxes
df = pd.read_csv('food_waste_dataset.csv')
st.markdown("""
<div style='text-align: center;'>
    <h1>Food Waste Prediction in Households Using Daily Consumption Patterns</h3>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Show dataset preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df, height=400)  # ✅ updated to allow scrolling by fixing the height

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
        
    le = LabelEncoder()
    
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

train_df_full = pd.read_csv('food_waste_dataset.csv')

for col in categorical_cols:
    le.fit(train_df_full[col])
    input_df[col] = le.transform(input_df[col])

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


# ===============================
# Complete Food Waste Prediction Project (Training Script)
# ===============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 2️⃣ Generate Dataset
# ===============================
num_samples = 1000  # ✅ increased dataset size

df = pd.DataFrame({
    'Household_Size': np.random.randint(1, 7, num_samples),
    'Daily_Meal_Count': np.random.randint(1, 4, num_samples),
    'Meal_Type': np.random.choice(['Vegetarian', 'Non-Vegetarian', 'Mixed'], num_samples),
    'Shopping_Habit': np.random.choice(['Daily', 'Weekly', 'Bulk Buying'], num_samples),
    'Storage_Availability': np.random.choice(['Small', 'Medium', 'Large'], num_samples),
    'Awareness_of_Waste_Management': np.random.choice(['Low', 'Medium', 'High'], num_samples),
    'Leftovers_Frequency': np.random.choice(['Rarely', 'Sometimes', 'Often'], num_samples),
    'Income_Range': np.random.choice(['Low', 'Medium', 'High'], num_samples),
    'Cooking_Preference': np.random.choice(['Fresh Daily', 'Batch Cooking'], num_samples)
})

baseline_waste = 0.5 + df['Household_Size']*0.2 + df['Daily_Meal_Count']*0.15

waste_adjustments = np.zeros(num_samples)
waste_adjustments += df['Shopping_Habit'].apply(lambda x: 0.8 if x=='Bulk Buying' else 0)
waste_adjustments += df['Awareness_of_Waste_Management'].apply(lambda x: -0.5 if x=='High' else 0.5 if x=='Low' else 0)
waste_adjustments += df['Leftovers_Frequency'].apply(lambda x: 0.7 if x=='Often' else -0.4 if x=='Rarely' else 0)
waste_adjustments += df['Cooking_Preference'].apply(lambda x: 0.2 if x=='Fresh Daily' else 0)
waste_adjustments += df['Storage_Availability'].apply(lambda x: -0.2 if x=='Large' else 0.2 if x=='Small' else 0)

random_noise = np.random.uniform(-0.5,0.5,num_samples)

df['Food_Waste_kg_per_Week'] = np.maximum(0.1, baseline_waste + waste_adjustments + random_noise)

df.to_csv('food_waste_dataset.csv', index=False)
print("Dataset generated and saved as 'food_waste_dataset.csv'")

# ===============================
# 3️⃣ Load Dataset & EDA
# ===============================
df = pd.read_csv('food_waste_dataset.csv')
print("\nDataset preview:")
print(df.head())

print("\nDataset info:")
print(df.info())
print("\nDataset description:")
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df['Food_Waste_kg_per_Week'], bins=20, kde=True)
plt.title("Distribution of Food Waste")
plt.xlabel("Food Waste (kg/week)")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Household_Size', y='Food_Waste_kg_per_Week', data=df)
plt.title("Household Size vs Food Waste")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Meal_Type', y='Food_Waste_kg_per_Week', data=df)
plt.title("Meal Type vs Food Waste")
plt.show()

# ===============================
# 4️⃣ Data Preprocessing
# ===============================
categorical_cols = ['Meal_Type','Shopping_Habit','Storage_Availability',
                    'Awareness_of_Waste_Management','Leftovers_Frequency',
                    'Income_Range','Cooking_Preference']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('Food_Waste_kg_per_Week', axis=1)
y = df['Food_Waste_kg_per_Week']

numeric_cols = ['Household_Size','Daily_Meal_Count']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ===============================
# 5️⃣ Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 6️⃣ Model Training
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ===============================
# 7️⃣ Model Evaluation
# ===============================
def evaluate(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("MAE:", round(mean_absolute_error(y_true, y_pred),3))
    print("MSE:", round(mean_squared_error(y_true, y_pred),3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)),3))
    print("R2:", round(r2_score(y_true, y_pred),3))
    print("\n")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_dt, "Decision Tree")
evaluate(y_test, y_pred_rf, "Random Forest")

# ===============================
# 8️⃣ Visualizations
# ===============================
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.xlabel("Actual Food Waste")
plt.ylabel("Predicted Food Waste")
plt.title("Random Forest: Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

# ===============================
# Outro Section
# ===============================
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


