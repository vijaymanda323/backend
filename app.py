from flask import Flask, request, jsonify
import joblib
import instaloader
import logging
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and scaler
MODEL_PATH = "fake_profile_model.pkl"
SCALER_PATH = "scaler.pkl"

model, scaler = None, None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Model and scaler loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading model/scaler: {e}")
else:
    logger.error("❌ Model or scaler file is missing. Please upload them.")

# Function to scrape Instagram profile details
def scrape_instagram_profile(username):
    loader = instaloader.Instaloader()
    
    try:
        logger.info(f"🔍 Fetching profile for: {username}")
        profile = instaloader.Profile.from_username(loader.context, username)
        
        profile_data = {
            "followers": profile.followers,
            "posts": profile.mediacount,
            "profile_pic": 1 if profile.profile_pic_url else 0,
            "description_length": len(profile.biography) if profile.biography else 0
        }

        logger.info(f"✅ Profile Data: {profile_data}")
        return profile_data

    except instaloader.exceptions.ProfileNotExistsException:
        logger.warning(f"⚠️ Profile '{username}' not found.")
        return {"error": "Profile not found"}
    except instaloader.exceptions.PrivateProfileNotFollowedException:
        logger.warning(f"🔒 Profile '{username}' is private.")
        return {"error": "Profile is private"}
    except instaloader.exceptions.InstaloaderException as e:
        logger.error(f"❌ Instaloader error: {e}")
        return {"error": str(e)}

@app.route('/')
def home():
    return "Flask backend is running on Render!"

@app.route("/detect", methods=["POST"])
def detect_fake_profile():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler file is missing. Please upload them."}), 500

    data = request.json
    username = data.get("username")

    if not username:
        return jsonify({"error": "Username is required"}), 400

    logger.info(f"🔍 Processing request for username: {username}")

    profile_data = scrape_instagram_profile(username)

    if "error" in profile_data:
        logger.warning(f"⚠️ {profile_data['error']}")
        return jsonify(profile_data), 404

    feature_values = [
        profile_data.get("followers", 0),
        profile_data.get("posts", 0),
        profile_data.get("profile_pic", 0),
        profile_data.get("description_length", 0)
    ]

    logger.info(f"🔮 Features for model: {feature_values}")

    try:
        features_array = np.array(feature_values).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        logger.info(f"🔄 Scaled Features: {features_scaled}")

        prediction = model.predict(features_scaled)[0]  # Predict real or fake
        logger.info(f"🧠 Model Prediction: {prediction}")

        return jsonify({
            "status": "real" if prediction == 0 else "fake",
            "profile_data": profile_data
        })
    except Exception as e:
        logger.error(f"❌ Prediction Error: {e}")
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)
