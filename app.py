from flask import Flask, request, jsonify
import joblib
import logging
import numpy as np
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and scaler
try:
    model = joblib.load("fake_profile_model.pkl")
    scaler = joblib.load("scaler.pkl")
    logger.info("✅ Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model/scaler: {e}")
    model, scaler = None, None

# Simulate Instagram profile details (NA values)
def simulate_instagram_profile(username):
    logger.info(f"🔧 Simulating profile data for username: {username}")
    
    profile_data = {
        "followers": "NA",
        "posts": "NA",
        "profile_pic": "NA",
        "description_length": "NA"
    }
    return profile_data

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake Profile Detector API is running 🛡️"}), 200

@app.route("/detect", methods=["POST"])
def detect_fake_profile():
    if not model or not scaler:
        logger.error("❌ Model or scaler not loaded. Cannot process detection.")
        return jsonify({"error": "Server model not loaded properly"}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "Request must be in JSON format"}), 400

    username = data.get("username")

    if not username:
        return jsonify({"error": "Username is required"}), 400

    logger.info(f"🔍 Processing profile: {username}")

    profile_data = simulate_instagram_profile(username)

    # Dummy feature values (since real scraping is not done)
    feature_values = [0, 0, 0, 0]

    logger.info(f"🔮 Features for model: {feature_values}")

    try:
        features_array = np.array(feature_values).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        logger.info(f"🔄 Scaled Features: {features_scaled}")

        prediction = model.predict(features_scaled)[0]  # 0 = real, 1 = fake

        logger.info(f"🧠 Model Prediction: {prediction}")

        return jsonify({
            "status": "real" if prediction == 0 else "fake",
            "profile_data": profile_data
        })

    except Exception as e:
        logger.error(f"❌ Prediction Error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT env variable
    app.run(host="0.0.0.0", port=port)
