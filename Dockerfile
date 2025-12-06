# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies required by OpenCV, including libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
