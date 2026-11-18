
# Python base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML model and application
COPY . .

# Expose Flask port
EXPOSE 5000

# Start Flask application
CMD ["python", "app.py"]
