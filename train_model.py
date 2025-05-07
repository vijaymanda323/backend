import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data(path):
    df = pd.read_csv(path)
    X = df[["#followers", "#posts", "profile pic", "description length"]]
    y = df["fake"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(max_depth=10, n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nFeature Importances:")
    for name, importance in zip(["#followers", "#posts", "profile pic", "description length"], model.feature_importances_):
        print(f"{name}: {importance:.4f}")

def save_artifacts(model, scaler):
    joblib.dump(model, "fake_profile_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nModel and scaler saved successfully!")

# === Pipeline ===
X_train, X_test, y_train, y_test = load_data(r"C:\Users\aparn\Downloads\test_final.csv")
model, scaler = train_model(X_train, y_train)
evaluate_model(model, scaler, X_test, y_test)
save_artifacts(model, scaler)


