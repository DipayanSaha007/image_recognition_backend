from flask import Flask, request, jsonify
from flask_cors import CORS
import util
import os
from PIL import Image
from io import BytesIO
import base64  # Make sure this is imported

app = Flask(__name__)
CORS(app, origins=["https://image-recognition-liard.vercel.app"])

@app.route('/')
def home():
    return 'Welcome to the Sports Celebrity Image Classification API'

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        # Parse the JSON request
        data = request.json
        image_data = data.get("image_data")
        if not image_data:
            return jsonify({"error": "Image data is missing"}), 400

        # Decode the base64 data (ensure it is correctly processed)
        image_bytes = base64.b64decode(image_data)
        # Additional processing logic...
        
        return jsonify(results)  # Replace with actual results
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
