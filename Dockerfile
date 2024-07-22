# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7001 available to the world outside this container
EXPOSE 7001

# Run streamlit when the container launches
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7001", "--server.address=0.0.0.0"]