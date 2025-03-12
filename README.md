# Face Extraction API with Flask and DeepFace

## Description

A simple Flask-based API that extracts the highest-confidence face from an image provided via URL, expands its bounding box by 20%, and returns the cropped image as a downloadable JPEG file without saving it on the server.

## Features

- Download an image from a given URL
- Detect faces using the DeepFace library (using the Yunet detector)
- Select the face with the highest confidence
- Expand the detected face region by 20%
- Return the cropped image as a downloadable attachment

## Requirements

- Python 3.9+
- Docker (optional, for containerized deployment)

## Installation and Running Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/face-extraction-api.git
   cd face-extraction-api
