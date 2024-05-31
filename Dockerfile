# Use Python image - wanted to use a slim version but netcat was not compatible
FROM python:3.10.6

# set working directory
WORKDIR /app

# Install gcc, netcat, curl and other dependencies
RUN apt-get update && apt-get install -y gcc python3-dev netcat curl && rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining application files
COPY . .

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Use the entrypoint script to trigger helper scripts and check weaviate availability
ENTRYPOINT ["/app/entrypoint.sh"]
