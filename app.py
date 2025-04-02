from flask import Flask, request, jsonify
import joblib
import instaloader
import logging
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instagram session details
INSTAGRAM_USERNAME = "_vijay.manda"  
SESSION_FILE = f"./sessions/session-{INSTAGRAM_USERNAME}"  # Ensure this is the correct session filename

# Load ML Model and Scaler
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

    # Load session file if available
    if os.path.exists(SESSION_FILE):
        try:
            loader.load_session_from_file(INSTAGRAM_USERNAME)  # No need for full path
            logger.info("✅ Successfully loaded Instagram session file!")
        except Exception as e:
            logger.error(f"❌ Error loading session file: {e}")

    try:
        profile = instaloader.Profile.from_username(loader.context, username)

        # Extract profile details
        profile_data = {
            "followers": profile.followers,
            "posts": profile.mediacount,
            "profile_pic": 1 if profile.profile_pic_url else 0,
            "description_length": len(profile.biography) if profile.biography else 0
        }

        logger.info(f"✅ Scraped Profile Data: {profile_data}")
        return profile_data

    except instaloader.exceptions.ProfileNotExistsException:
        logger.warning(f"⚠️ Profile '{username}' not found.")
        return {"error": "Profile not found"}
    except instaloader.exceptions.PrivateProfileNotFollowedException:
        logger.warning(f"🔒 Profile '{username}' is private.")
        return {"error": "Profile is private"}
    except instaloader.exceptions.InstaloaderException as e:
        if "Please wait a few minutes" in str(e):
            return {"error": "Instagram is blocking requests. Try again later."}
        logger.error(f"❌ Instaloader error: {e}")
        return {"error": str(e)}

@app.route('/')
def home():
    return "Flask backend is running!"

@app.route("/detect", methods=["POST"])
def detect_fake_profile():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler file is missing. Please upload them."}), 500

    try:
        data = request.get_json()
        if not data or "username" not in data:
            return jsonify({"error": "Username is required"}), 400
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    username = data["username"]
    logger.info(f"🔍 Fetching profile: {username}")

    profile_data = scrape_instagram_profile(username)

    if not profile_data or "error" in profile_data:
        logger.warning("⚠️ Profile not found or private")
        return jsonify(profile_data), 404

    logger.info(f"✅ Profile Data: {profile_data}")

    # Extract correct features for prediction
    feature_values = [
        profile_data.get("followers", 0),
        profile_data.get("posts", 0),
        profile_data.get("profile_pic", 0),
        profile_data.get("description_length", 0)
    ]

    logger.info(f"🔮 Features for model: {feature_values}")

    try:
        # Ensure correct shape for prediction & apply scaling
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
