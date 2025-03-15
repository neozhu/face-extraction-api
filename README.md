[![Docker Build and Push](https://github.com/neozhu/face-extraction-api/actions/workflows/docker-image.yml/badge.svg)](https://github.com/neozhu/face-extraction-api/actions/workflows/docker-image.yml)  
[![Python Compile Check](https://github.com/neozhu/face-extraction-api/actions/workflows/python-app.yml/badge.svg)](https://github.com/neozhu/face-extraction-api/actions/workflows/python-app.yml)

# Face Extraction API with Flask and DeepFace

## Description

A simple Flask-based API that extracts the highest-confidence face from an image, expands its bounding box by 20%, and returns the cropped image as a downloadable JPEG file without saving it on the server. The API supports two methods of image input: providing an image URL or uploading an image file.

## Features

- **Image URL Input**: Downloads an image from a given URL.
- **File Upload**: Accepts an image file uploaded via a form-data POST request.
- Detects faces using the DeepFace library with the Yunet detector.
- Selects the face with the highest confidence.
- Expands the detected face region by 20%.
- Returns the cropped image as a downloadable attachment.
- **Minio Integration**: Two additional endpoints allow uploading the extracted face (or the original image with the cropped face) directly to a Minio storage server.

## Example

The following demonstrates the original image and the cropped face image side by side:

<p>
  <img src="img/original_image.jpg" alt="Original Image" width="45%" style="display: inline-block; margin-right: 5%;" />
  <img src="img/extracted_face.jpg" alt="Cropped Face Image" width="41%" style="display: inline-block;" />
</p>

## Requirements

- Python 3.9+
- Docker (optional, for containerized deployment)

## Installation and Running Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/face-extraction-api.git
   cd face-extraction-api
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Minio configuration. See the example below:
   ```ini
   # .env Example File
   MINIO_ENDPOINT=minio.blazorserver.com
   MINIO_ACCESS_KEY=your_minio_access_key
   MINIO_SECRET_KEY=your_minio_secret_key
   MINIO_BUCKET_NAME=your_bucket_name
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`.

## Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t face-extraction-api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 face-extraction-api
   ```

3. Docker compose file:
   ```yml
   version: '3.8'
   services:
     face-extraction-api:
       image: blazordevlab/face-extraction-api:latest
       container_name: faceextraction
       ports:
         - "5000:5000"
       labels:
         - "traefik.enable=true"
         # Replace 'faceextraction.example.com' with your actual domain
         - "traefik.http.routers.faceextraction.rule=Host(`faceextraction.example.com`)"
         - "traefik.http.routers.faceextraction.tls=true"
         - "traefik.http.routers.faceextraction.entrypoints=https"
         - "traefik.http.services.faceextraction.loadbalancer.server.port=5000"
         - "traefik.docker.network=proxy"
       networks:
         proxy:
       security_opt:
         - no-new-privileges:true
   networks:
      proxy:
         external: true      
   ```

## API Endpoints

### 1. Extract Face by Image URL

- **Endpoint**: `/extract_face`
- **Method**: POST
- **Payload**: JSON object containing the `image_url` key.
  
  **Example JSON Payload**:
  ```json
  {
    "image_url": "https://example.com/your_image.jpg"
  }
  ```

- **Response**: Returns the cropped face image as a downloadable JPEG file.

### 2. Extract Face by File Upload

- **Endpoint**: `/upload_extract_face`
- **Method**: POST
- **Payload**: Form-data with an image file in the `file` field.
  
  **Example using curl**:
  ```bash
  curl -X POST -F "file=@/path/to/your_image.jpg" http://localhost:5000/upload_extract_face --output extracted_face.jpg
  ```

- **Response**: Returns the cropped face image as a downloadable JPEG file.

### 3. Extract Face and Upload by Image URL to Minio

- **Endpoint**: `/extract_face_to_minio`
- **Method**: POST
- **Payload**: JSON object containing the `image_url` key.
  
  **Example JSON Payload**:
  ```json
  {
    "image_url": "https://example.com/your_image.jpg"
  }
  ```

- **Response**: Returns a JSON response with:
  - `status`: Operation status.
  - `message`: Confirmation that the face has been extracted and uploaded.
  - `url`: The full URL of the uploaded cropped face image.
  - `source`: Indicates whether the source URL was from the Minio server or external.

### 4. Upload Image and Extract Face to Minio

- **Endpoint**: `/upload_face_to_minio`
- **Method**: POST
- **Payload**: Form-data with:
  - an image file in the `file` field,
  - a `path` parameter specifying the target directory in Minio.
  
  **Example using curl**:
  ```bash
  curl -X POST -F "file=@/path/to/your_image.jpg" -F "path=test_upload" http://localhost:5000/upload_face_to_minio
  ```

- **Response**: Returns a JSON response with:
  - `status`: Operation status.
  - `message`: Confirmation that the file was uploaded to Minio successfully.
  - `uploaded_to`: Indicates that the file was uploaded to Minio.
  - `original_url`: The full URL of the original image.
  - `face_url`: The full URL of the cropped face image (saved under a `faces/` subdirectory).

## License

This project is licensed under the MIT License.

