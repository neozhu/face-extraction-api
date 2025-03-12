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

3. Run the Flask application:
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

## License

This project is licensed under the MIT License.
