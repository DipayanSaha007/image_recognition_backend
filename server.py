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
    data = request.get_json()
    encoded_image = data.get('image_data')  # Retrieve image data

    if not encoded_image:
        return jsonify({"error": "No image_data provided"}), 400

    try:
        # Ensure we handle base64 strings correctly
        if ',' in encoded_image:
            base64_image = encoded_image.split(',')[1]  # Remove metadata (data:image/png;base64,)
        else:
            base64_image = encoded_image

        # Decode the base64 image string
        decoded_image = base64.b64decode(base64_image)
        
        # Convert binary data to an image object
        image = Image.open(BytesIO(decoded_image))
        
        # Use your utility function to classify the image
        classification_result = util.classify_image(encoded_image)  # Pass the original base64 string if needed

        response = jsonify(classification_result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
