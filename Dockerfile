# Use an official PyTorch base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file and server script
COPY model.pth .
COPY model_server.py .

# Create the templates directory and copy index.html into it
RUN mkdir templates
COPY index.html templates/

# Create the templates directory and copy index.html into it
RUN mkdir static
COPY styles.css static/
COPY favicon.ico static/

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "model_server.py"]
