import joblib
import json
import numpy as np
import cv2
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(decoded_image_data, file_path=None):
    """
    Classify the image using the pre-trained model.
    :param decoded_image_data: Decoded binary image data (from base64).
    :param file_path: Path to the image file (optional, alternative to binary data).
    :return: List of classification results with probabilities.
    """
    imgs = get_cropped_image_if_2_eyes(file_path, decoded_image_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    """
    Convert a class number to its corresponding name.
    :param class_num: Class number.
    :return: Class name.
    """
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    """
    Load the saved model and class dictionaries.
    """
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_cv2_image_from_binary_data(binary_data):
    """
    Convert binary image data into an OpenCV-compatible image.
    :param binary_data: Decoded binary image data.
    :return: OpenCV image.
    """
    nparr = np.frombuffer(binary_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_data):
    """
    Detect faces and eyes in the image and return cropped regions containing faces with at least 2 eyes.
    :param image_path: File path to the image.
    :param image_data: Decoded binary image data (optional alternative to file path).
    :return: List of cropped face images.
    """
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_binary_data(image_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_virat():
    """
    Utility function to read a base64 test image (example usage).
    :return: Base64-encoded string of a test image.
    """
    with open("./b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()

    # Test cases (uncomment for debugging)
    # print(classify_image(get_b64_test_image_for_virat(), None))
    # print(classify_image(None, "./test_images/federer1.jpg"))
    # print(classify_image(None, "./test_images/virat1.jpg"))
