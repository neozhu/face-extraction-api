import cv2
import numpy as np
import requests
import io
import os
import urllib.parse
from flask import Flask, request, jsonify, send_file
from deepface import DeepFace
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Minio configuration from environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio.blazorserver.com")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

# Initialize Flask app
app = Flask(__name__)

# Initialize Minio client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True  # Set to False if not using HTTPS
)

# Ensure bucket exists
def ensure_bucket_exists():
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
            minio_client.make_bucket(MINIO_BUCKET_NAME)
    except S3Error as e:
        app.logger.error(f"Error checking/creating bucket: {e}")
        raise

@app.route('/', methods=['GET'])
def index():
    return "Application started successfully", 200

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

@app.route('/extract_face_to_minio', methods=['POST'])
def extract_face_to_minio():
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
    
    # Process the image and upload to Minio
    try:
        # Get processed face image as bytes
        face_jpeg = process_image_to_bytes(img)
        if not face_jpeg:
            return jsonify({"error": "Image processing failed"}), 500
        
        # Parse the URL
        parsed_url = urllib.parse.urlparse(image_url)
        
        # Check if the URL is from our own Minio server
        is_from_own_minio = parsed_url.netloc == MINIO_ENDPOINT
        
        # Determine the object name based on URL source
        if is_from_own_minio:
            # If URL is from our Minio, extract the path and determine if it contains bucket name
            path = parsed_url.path.lstrip('/')
            # Check if path starts with bucket name
            if path.startswith(f"{MINIO_BUCKET_NAME}/"):
                # Remove bucket name from path
                path = path[len(f"{MINIO_BUCKET_NAME}/"):]
            # Insert 'faces/' after the first directory if it exists
            path_parts = path.split('/', 1)
            if len(path_parts) > 1:
                object_name = f"{path_parts[0]}/faces/{path_parts[1]}"
            else:
                object_name = f"faces/{path}"
        else:
            # For external URLs, just put in faces/ directory
            # Extract path from original URL
            url_path = parsed_url.path.lstrip('/')
            # For external URLs, just use the filename part
            filename = os.path.basename(url_path)
            object_name = f"faces/{filename}"
        
        # Ensure the bucket exists
        ensure_bucket_exists()
        
        # Upload to Minio
        minio_client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=object_name,
            data=io.BytesIO(face_jpeg),
            length=len(face_jpeg),
            content_type="image/jpeg"
        )
        
        # Generate the full URL to the uploaded image
        minio_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET_NAME}/{object_name}"
        
        return jsonify({
            "status": "success",
            "message": "Face extracted and uploaded to Minio",
            "url": minio_url,
            "source": "own_minio" if is_from_own_minio else "external"
        })
        
    except S3Error as e:
        return jsonify({"error": f"Minio upload failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# New endpoint: upload_face_to_minio
# This endpoint accepts an uploaded image file and a 'path' parameter,
# saves the original image to Minio at the specified path,
# crops the face from the image, and saves it under a 'faces/' subdirectory.
# It returns the full URLs for both the original image and the cropped face image,
# along with a note that the file is uploaded to Minio.
@app.route('/upload_face_to_minio', methods=['POST'])
def upload_face_to_minio():
    # Check if the 'file' parameter exists in the request
    if 'file' not in request.files:
        return jsonify({"error": "Missing file parameter in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Get the 'path' parameter from form data
    upload_path = request.form.get("path")
    if not upload_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    # Read the uploaded file and decode the image
    try:
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decoding failed")
    except Exception as e:
        return jsonify({"error": f"Failed to read or decode uploaded file: {str(e)}"}), 500

    # Ensure the Minio bucket exists
    try:
        ensure_bucket_exists()
    except Exception as e:
        return jsonify({"error": "Bucket check/creation failed", "details": str(e)}), 500

    # Construct object names for the original and cropped face images
    original_object_name = f"{upload_path.rstrip('/')}/{file.filename}"
    face_object_name = f"{upload_path.rstrip('/')}/faces/{file.filename}"

    # Upload the original image to Minio
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=original_object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=file.content_type or "application/octet-stream"
        )
    except S3Error as e:
        return jsonify({"error": f"Original image upload to Minio failed: {str(e)}"}), 500

    # Process the image to extract the face
    face_bytes = process_image_to_bytes(img)
    if face_bytes is None:
        return jsonify({"error": "Face detection or cropping failed"}), 500

    # Upload the cropped face image to Minio
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=face_object_name,
            data=io.BytesIO(face_bytes),
            length=len(face_bytes),
            content_type="image/jpeg"
        )
    except S3Error as e:
        return jsonify({"error": f"Cropped face image upload to Minio failed: {str(e)}"}), 500

    # Construct full URLs for the original and face images
    original_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET_NAME}/{original_object_name}"
    face_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET_NAME}/{face_object_name}"

    return jsonify({
        "status": "success",
        "message": "File uploaded to Minio successfully",
        "uploaded_to": "minio",
        "original_url": original_url,
        "face_url": face_url
    }), 200

def process_image(img):
    # Process the image and return the result as a file
    face_jpeg = process_image_to_bytes(img)
    if not face_jpeg:
        return jsonify({"error": "Image processing failed"}), 500
    
    # Return the cropped image as an attachment to the client
    return send_file(
        io.BytesIO(face_jpeg),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name="extracted_face.jpg"
    )

def process_image_to_bytes(img):
    # Get the dimensions of the original image
    img_height, img_width = img.shape[:2]
    
    # Extract faces using the yunet model
    try:
        faces = DeepFace.extract_faces(img, detector_backend="yunet", enforce_detection=True)
    except Exception as e:
        app.logger.error(f"Face detection failed: {str(e)}")
        return None
    
    if len(faces) == 0:
        app.logger.error("No face detected")
        return None
    
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
        app.logger.error("Image encoding failed")
        return None
    
    return jpeg.tobytes()

if __name__ == '__main__':
    # Note: When running the Flask server in a Jupyter Notebook,
    # it is recommended to set debug=False or use threading mode to avoid blocking the Notebook.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
