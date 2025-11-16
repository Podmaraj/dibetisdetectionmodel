# ===============================
# 📦 Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ===============================
# 🧹 Load and Preprocess Data
# ===============================
df = pd.read_csv("diabetes.csv")

# Add a simulated gender column (1 = Male, 0 = Female)
np.random.seed(42)
df["Gender"] = np.random.choice([0, 1], size=len(df))

# Handle missing or zero values
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.drop(columns='Insulin', inplace=True)
df.fillna(df.median(), inplace=True)

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance the data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ===============================
# 🧠 Define Models
# ===============================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ===============================
# ⚙️ Train and Evaluate All Models
# ===============================
results = []
plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    results.append([name, acc, prec, roc_auc])

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of All Models")
plt.legend()
plt.show()

# ===============================
# 📊 Compare Results
# ===============================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "AUC"])
results_df = results_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)
print("📊 Model Comparison Results:\n")
print(results_df)

# ===============================
# 🏆 Train Best Model (Gradient Boosting Fine-Tuned)
# ===============================
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

gb = GradientBoostingClassifier(random_state=42)
grid = GridSearchCV(gb, params, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("✅ Best Parameters:", grid.best_params_)

best_gb = grid.best_estimator_
y_pred = best_gb.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\n🎯 Final Trained Model: Gradient Boosting with Tuned Parameters")
