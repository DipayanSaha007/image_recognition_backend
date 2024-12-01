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
