# ==============================
# 📦 Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 🧹 Step 1: Load and Prepare Data
# ==============================
df = pd.read_csv('diabetes.csv')

# Add Gender column randomly (1 = Male, 0 = Female)
np.random.seed(42)
df["Gender"] = np.random.choice([0, 1], size=len(df))

# ✅ If Gender is Male → Pregnancies = 0
df.loc[df["Gender"] == 1, "Pregnancies"] = 0

# Replace 0 with NaN in certain columns
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Drop 'Insulin' column (too many missing)
df.drop(columns='Insulin', inplace=True)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Define features and label
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Balance using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ==============================
# 🧠 Step 2: Train Model (XGBoost)
# ==============================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_res, y_train_res)

# ==============================
# 📊 Step 3: Evaluate
# ==============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n📊 Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 🧍 Step 4: User Input
# ==============================
print("\n🔹 Enter Patient Details to Predict Diabetes 🔹")

Gender_input = input("Enter Gender (Male/Female): ").strip().lower()
if Gender_input == 'male':
    Gender = 1
    Pregnancies = 0   # Automatically set for male
else:
    Gender = 0
    Pregnancies = float(input("Enter Number of Pregnancies: "))

Glucose = float(input("Enter Glucose Level: "))
BloodPressure = float(input("Enter Blood Pressure: "))
SkinThickness = float(input("Enter Skin Thickness: "))
BMI = float(input("Enter BMI: "))
DiabetesPedigreeFunction = float(input("Enter Diabetes Pedigree Function: "))
Age = float(input("Enter Age: "))

# Create input DataFrame
user_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, BMI,
                           DiabetesPedigreeFunction, Age, Gender]],
                         columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                  'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Gender'])

# Scale input
user_data_scaled = scaler.transform(user_data)

# Predict
prediction = model.predict(user_data_scaled)[0]
probability = model.predict_proba(user_data_scaled)[0][1]

# ==============================
# 🩺 Step 5: Display Result
# ==============================
print("\n===============================")
if prediction == 1:
    print(f"🩸 The person is LIKELY to have Diabetes (Confidence: {probability*100:.2f}%)")
else:
    print(f"✅ The person is NOT likely to have Diabetes (Confidence: {(1-probability)*100:.2f}%)")
print("===============================\n")
import pickle
# Save model and scaler
with open("diabetes_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
print("✅ Model and Scaler saved successfully as 'diabetes_model.pkl' and 'scaler.pkl'")

