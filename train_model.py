import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_excel("customer_churn_categorical.xlsx")
print(df.columns.tolist())

# Preprocess
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['Gender','Subscription Type','Contract Length'], drop_first=True)

# Select features based on your UI
feature_cols = [col for col in df.columns if col not in ['CustomerID','Churn']]

X = df[feature_cols]
y = df['Churn']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'churn_model.pkl')
print("Model trained and saved.")
