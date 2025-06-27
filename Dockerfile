FROM python:3.11-slim-buster

# Step 1: Install libgomp1 for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy code into the container
COPY . /app

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Start the app
CMD ["python3", "app.py"]
