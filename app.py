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
try:
    model = joblib.load("fake_profile_model.pkl")
    scaler = joblib.load("scaler.pkl")
    logger.info("✅ Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model/scaler: {e}")
    model, scaler = None, None

# Instagram session details
INSTAGRAM_USERNAME = "_vijay.manda"  # Replace with your Instagram username
SESSION_FILE = os.path.join(os.getcwd(), "session-instagram")

# Function to scrape Instagram profile details
def scrape_instagram_profile(username):
    loader = instaloader.Instaloader()

    try:
        # Load the Instaloader session
        loader.load_session_from_file(INSTAGRAM_USERNAME, SESSION_FILE)

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
        return None
    except instaloader.exceptions.PrivateProfileNotFollowedException:
        logger.warning(f"🔒 Profile '{username}' is private.")
        return {"error": "Profile is private"}
    except instaloader.exceptions.InstaloaderException as e:
        logger.error(f"❌ Instaloader error: {e}")
        return None
@app.route('/')
def home():
    return "Flask backend is running!"

@app.route("/detect", methods=["POST"])
def detect_fake_profile():
    data = request.json
    username = data.get("username")

    if not username:
        return jsonify({"error": "Username is required"}), 400

    logger.info(f"🔍 Fetching profile: {username}")

    profile_data = scrape_instagram_profile(username)

    if not profile_data or "error" in profile_data:
        logger.warning("⚠️ Profile not found or private")
        return jsonify({"error": "Profile not found or is private"}), 404

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