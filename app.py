from flask import Flask, request, jsonify
import joblib
import instaloader
import logging
import numpy as np
from flask_cors import CORS

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

# Instagram session details
INSTAGRAM_USERNAME = "_vijay.manda"  # Replace with your Instagram username
SESSION_FILE = "C:/Users/aparn/AppData/Local/Instaloader/session-_vijay.manda"

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
    except Exception as e:
        logger.error(f"❌ Unexpected scraping error: {e}")
        return None

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

    logger.info(f"🔍 Fetching profile: {username}")

    profile_data = scrape_instagram_profile(username)

    if not profile_data:
        return jsonify({"error": "Profile not found or could not be fetched"}), 404
    if "error" in profile_data:
        return jsonify(profile_data), 403  # Private profile

    # Prepare features for prediction
    feature_values = [
        profile_data.get("followers", 0),
        profile_data.get("posts", 0),
        profile_data.get("profile_pic", 0),
        profile_data.get("description_length", 0)
    ]

    logger.info(f"🔮 Features for model: {feature_values}")

    try:
        # Ensure correct shape for prediction and apply scaling
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
    app.run(host='0.0.0.0', port=8000)
