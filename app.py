# First, ensure you have installed the required dependencies:
# pip install flask deepface opencv-python requests matplotlib

import cv2
import numpy as np
import requests
import io
from flask import Flask, request, jsonify, send_file
from deepface import DeepFace

app = Flask(__name__)

@app.route('/extract_face', methods=['POST'])
def extract_face():
    # Get the image URL from the request
    data = request.get_json()
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Missing image_url parameter"}), 400

    # Download and decode the image
    try:
        response = requests.get(image_url)
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Failed to download or decode image: {str(e)}"}), 500

    return process_image(img)

@app.route('/upload_extract_face', methods=['POST'])
def upload_extract_face():
    # Check if the file part is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Read and decode the uploaded image file
    try:
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Failed to read uploaded file: {str(e)}"}), 500

    return process_image(img)

def process_image(img):
    # Get the dimensions of the original image
    img_height, img_width = img.shape[:2]

    # Extract faces using the yunet model (new method)
    try:
        faces = DeepFace.extract_faces(img, detector_backend="yunet", enforce_detection=True)
    except Exception as e:
        return jsonify({"error": f"Face detection failed: {str(e)}"}), 500

    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 404

    # Select the face with the highest confidence
    best_face = max(faces, key=lambda face: face.get("confidence", 0))
    facial_area = best_face['facial_area']
    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

    # Expand the face region by 20%
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, img_width)
    y2 = min(y + h + margin_y, img_height)

    # Crop the expanded face region
    expanded_face = img[y1:y2, x1:x2]

    # Encode the cropped image as JPEG
    ret, jpeg = cv2.imencode('.jpg', expanded_face)
    if not ret:
        return jsonify({"error": "Image encoding failed"}), 500

    # Return the cropped image as an attachment to the client without saving it on the server
    return send_file(
        io.BytesIO(jpeg.tobytes()),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name="extracted_face.jpg"
    )

if __name__ == '__main__':
    # Note: When running the Flask server in a Jupyter Notebook,
    # it is recommended to set debug=False or use threading mode to avoid blocking the Notebook.
    app.run(port=5000, debug=True, use_reloader=False)
