# Dockerfile
FROM python:3.9-slim

# Create app directory
WORKDIR /app

# Copy your code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Generate Python gRPC stubs from inference.proto
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

# Expose ports (HTTP: 8080, gRPC: 9090)
EXPOSE 8080
EXPOSE 9090

# Supervisord config
RUN pip install supervisor
CMD [ "supervisord", "-c", "supervisord.conf" ]