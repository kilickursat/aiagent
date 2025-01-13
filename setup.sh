#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Set up Hugging Face token (replace with your actual token)
export HUGGINGFACE_TOKEN="api_key"

# Initialize Hugging Face login
python -c "from huggingface_hub import login; login('$HUGGINGFACE_TOKEN')"

# Create necessary directories
mkdir -p templates

# Move the index.html file to the templates directory
mv index.html templates/

# Set up logging directory
mkdir logs

# Set environment variables (adjust as needed)
export FLASK_APP=app.py
export FLASK_ENV=production

# Start the Flask application
gunicorn app:app
