import cv2
import numpy as np
import requests
import io
import os
import urllib.parse
import logging
from flask import Flask, request, jsonify, send_file
from deepface import DeepFace
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Minio configuration from environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio.blazorserver.com")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

# Initialize Flask app
app = Flask(__name__)

# Validate required environment variables
if not all([MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME]):
    logger.error("Missing required Minio configuration. Please check environment variables.")
    raise ValueError("Missing required Minio configuration")

# Initialize Minio client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True  # Set to False if not using HTTPS
    )
    logger.info(f"Minio client initialized successfully for endpoint: {MINIO_ENDPOINT}")
except Exception as e:
    logger.error(f"Failed to initialize Minio client: {e}")
    raise

# Ensure bucket exists
def ensure_bucket_exists():
    """Check if the bucket exists and create it if necessary."""
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
            minio_client.make_bucket(MINIO_BUCKET_NAME)
            logger.info(f"Created bucket: {MINIO_BUCKET_NAME}")
        else:
            logger.debug(f"Bucket already exists: {MINIO_BUCKET_NAME}")
    except S3Error as e:
        logger.error(f"Error checking/creating bucket: {e}")
        raise

def download_and_decode_image(image_url):
    """Download and decode an image from URL.
    
    Args:
        image_url: URL of the image to download
        
    Returns:
        Decoded image as numpy array or None if failed
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            return None
        return img
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while downloading image from {image_url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        return None
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    """Health check endpoint."""
    return jsonify({"status": "running", "message": "Face Extraction API"}), 200

@app.route('/extract_face', methods=['POST'])
def extract_face():
    """Extract face from an image URL and return the cropped face image."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Missing image_url parameter"}), 400
    
    logger.info(f"Processing face extraction from URL: {image_url}")
    
    # Download and decode the image
    img = download_and_decode_image(image_url)
    if img is None:
        return jsonify({"error": "Failed to download or decode image"}), 500
    
    return process_image(img)

@app.route('/upload_extract_face', methods=['POST'])
def upload_extract_face():
    """Extract face from an uploaded image file and return the cropped face image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    logger.info(f"Processing uploaded file: {file.filename}")
    
    # Read and decode the uploaded image file
    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Empty file uploaded"}), 400
            
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image. Please upload a valid image file."}), 400
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        return jsonify({"error": f"Failed to read uploaded file: {str(e)}"}), 500
    
    return process_image(img)

def generate_face_object_name(image_url):
    """Generate the object name for the face image based on the source URL.
    
    Args:
        image_url: Source image URL
        
    Returns:
        Tuple of (object_name, is_from_own_minio)
    """
    parsed_url = urllib.parse.urlparse(image_url)
    is_from_own_minio = parsed_url.netloc == MINIO_ENDPOINT
    
    if is_from_own_minio:
        # If URL is from our Minio, extract the path and determine if it contains bucket name
        path = parsed_url.path.lstrip('/')
        
        # Remove bucket name from path if present
        if path.startswith(f"{MINIO_BUCKET_NAME}/"):
            path = path[len(f"{MINIO_BUCKET_NAME}/"):]
        
        # Insert 'faces/' after the first directory if it exists
        path_parts = path.split('/', 1)
        if len(path_parts) > 1:
            object_name = f"{path_parts[0]}/faces/{path_parts[1]}"
        else:
            object_name = f"faces/{path}"
    else:
        # For external URLs, just use the filename in faces/ directory
        filename = os.path.basename(parsed_url.path.lstrip('/'))
        if not filename:
            filename = "extracted_face.jpg"
        object_name = f"faces/{filename}"
    
    return object_name, is_from_own_minio

@app.route('/extract_face_to_minio', methods=['POST'])
def extract_face_to_minio():
    """Extract face from an image URL and upload to Minio storage."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Missing image_url parameter"}), 400
    
    logger.info(f"Extracting face from URL and uploading to Minio: {image_url}")
    
    # Download and decode the image
    img = download_and_decode_image(image_url)
    if img is None:
        return jsonify({"error": "Failed to download or decode image"}), 500
    
    # Process the image to extract face
    face_jpeg = process_image_to_bytes(img)
    if not face_jpeg:
        return jsonify({"error": "Face detection failed. No face found in the image."}), 400
    
    # Generate object name based on source URL
    try:
        object_name, is_from_own_minio = generate_face_object_name(image_url)
    except Exception as e:
        logger.error(f"Error generating object name: {e}")
        return jsonify({"error": "Failed to generate storage path"}), 500
    
    # Ensure the bucket exists
    try:
        ensure_bucket_exists()
    except Exception as e:
        logger.error(f"Bucket validation failed: {e}")
        return jsonify({"error": "Storage bucket validation failed"}), 500
    
    # Upload to Minio
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=object_name,
            data=io.BytesIO(face_jpeg),
            length=len(face_jpeg),
            content_type="image/jpeg"
        )
        logger.info(f"Successfully uploaded face image to: {object_name}")
    except S3Error as e:
        logger.error(f"Minio upload failed: {e}")
        return jsonify({"error": f"Minio upload failed: {str(e)}"}), 500
    
    # Generate the full URL to the uploaded image
    minio_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET_NAME}/{object_name}"
    
    return jsonify({
        "status": "success",
        "message": "Face extracted and uploaded to Minio",
        "url": minio_url,
        "source": "own_minio" if is_from_own_minio else "external"
    }), 200

@app.route('/upload_face_to_minio', methods=['POST'])
def upload_face_to_minio():
    """Upload an image file to Minio and extract the face.
    
    This endpoint accepts an uploaded image file and a 'path' parameter,
    saves the original image to Minio at the specified path,
    crops the face from the image, and saves it under a 'faces/' subdirectory.
    Returns the full URLs for both the original image and the cropped face image.
    """
    # Validate file presence
    if 'file' not in request.files:
        return jsonify({"error": "Missing file parameter in request"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Validate path parameter
    upload_path = request.form.get("path")
    if not upload_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400
    
    logger.info(f"Uploading file '{file.filename}' to path '{upload_path}'")

    # Read and decode the uploaded file
    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Empty file uploaded"}), 400
            
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image. Please upload a valid image file."}), 400
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        return jsonify({"error": f"Failed to read or decode uploaded file: {str(e)}"}), 500

    # Ensure the Minio bucket exists
    try:
        ensure_bucket_exists()
    except Exception as e:
        logger.error(f"Bucket validation failed: {e}")
        return jsonify({"error": "Bucket check/creation failed", "details": str(e)}), 500

    # Construct object names for the original and cropped face images
    upload_path_clean = upload_path.rstrip('/')
    original_object_name = f"{upload_path_clean}/{file.filename}"
    face_object_name = f"{upload_path_clean}/faces/{file.filename}"

    # Upload the original image to Minio
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=original_object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=file.content_type or "application/octet-stream"
        )
        logger.info(f"Original image uploaded: {original_object_name}")
    except S3Error as e:
        logger.error(f"Failed to upload original image: {e}")
        return jsonify({"error": f"Original image upload to Minio failed: {str(e)}"}), 500

    # Process the image to extract the face
    face_bytes = process_image_to_bytes(img)
    if face_bytes is None:
        return jsonify({"error": "Face detection failed. No face found in the image."}), 400

    # Upload the cropped face image to Minio
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=face_object_name,
            data=io.BytesIO(face_bytes),
            length=len(face_bytes),
            content_type="image/jpeg"
        )
        logger.info(f"Face image uploaded: {face_object_name}")
    except S3Error as e:
        logger.error(f"Failed to upload face image: {e}")
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
    """Process an image to extract face and return as a file response.
    
    Args:
        img: OpenCV image (numpy array)
        
    Returns:
        Flask response with the cropped face image or error JSON
    """
    face_jpeg = process_image_to_bytes(img)
    if not face_jpeg:
        return jsonify({"error": "Face detection failed. No face found in the image."}), 400
    
    # Return the cropped image as an attachment to the client
    return send_file(
        io.BytesIO(face_jpeg),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name="extracted_face.jpg"
    )

def process_image_to_bytes(img):
    """Extract and crop the best face from an image.
    
    Args:
        img: OpenCV image (numpy array)
        
    Returns:
        JPEG-encoded face image as bytes, or None if face detection fails
    """
    if img is None:
        logger.error("Invalid image provided")
        return None
    
    # Get the dimensions of the original image
    img_height, img_width = img.shape[:2]
    
    # Extract faces using the yunet model
    try:
        faces = DeepFace.extract_faces(
            img, 
            detector_backend="yunet", 
            enforce_detection=True
        )
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        return None
    
    if not faces or len(faces) == 0:
        logger.warning("No face detected in the image")
        return None
    
    # Select the face with the highest confidence
    best_face = max(faces, key=lambda face: face.get("confidence", 0))
    confidence = best_face.get("confidence", 0)
    logger.info(f"Detected face with confidence: {confidence:.2f}")
    
    facial_area = best_face['facial_area']
    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
    
    # Expand the face region by 10% for better framing
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, img_width)
    y2 = min(y + h + margin_y, img_height)
    
    # Crop the expanded face region
    expanded_face = img[y1:y2, x1:x2]
    
    if expanded_face.size == 0:
        logger.error("Cropped face region is empty")
        return None
    
    # Encode the cropped image as JPEG with quality optimization
    ret, jpeg = cv2.imencode('.jpg', expanded_face, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        logger.error("Image encoding failed")
        return None
    
    return jpeg.tobytes()

# Ensure bucket exists on startup
try:
    ensure_bucket_exists()
    logger.info("Bucket validation successful")
except Exception as e:
    logger.error(f"Failed to validate bucket on startup: {e}")

if __name__ == '__main__':
    # This block is only used for local development
    # In production, Gunicorn will run the app directly
    logger.info("Starting Face Extraction API server on port 5000 (development mode)")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
