from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\aparn\Downloads\test_final.csv")

# Define features and target (using correct column names)
features = df[["#followers", "#posts", "profile pic", "description length"]]
labels = df["fake"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(max_depth=10, n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Save Model & Scaler
joblib.dump(model, "fake_profile_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained & saved successfully!")
