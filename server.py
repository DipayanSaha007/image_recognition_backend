from flask import Flask, request, jsonify
from flask_cors import CORS
import util
import os

app = Flask(__name__)
CORS(app, origins=["https://image-recognition-liard.vercel.app"])

@app.route('/')
def home():
    return 'Welcome to the Sports Celebrity Image Classification API'

@app.route('/classify_image', methods=['POST'])
def classify_image():
    # Use get_json() to retrieve JSON data from the request
    data = request.get_json()
    image_data = data.get('image_data')  # Retrieve image data

    if not image_data:
        return jsonify({"error": "No image_data provided"}), 400

    try:
        # Decode the base64 image data
        image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," part
        image_data = base64.b64decode(image_data)

        # Convert the binary data to an image
        image = Image.open(BytesIO(image_data))
        
        # Now you can use the image (e.g., classify it using your model)
        response = jsonify(util.classify_image(image))
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()

    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app with the host as 0.0.0.0 to allow external access
    app.run(host='0.0.0.0', port=port)
