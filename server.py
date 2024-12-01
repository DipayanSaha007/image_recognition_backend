from flask import Flask, request, jsonify
from flask_cors import CORS
import util
import os

app = Flask(__name__)
CORS(app)

@app.route('/classify_image', methods=['POST'])
def classify_image():
    image_data = request.form.get('image_data')
    if not image_data:
        return jsonify({"error": "No image_data provided"}), 400

    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()

    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app with the host as 0.0.0.0 to allow external access
    app.run(host='0.0.0.0', port=port)
