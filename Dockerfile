# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
#FROM python:3.9.0

# Install linux packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0 \
    libssl-dev \
    libffi-dev  \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    zip\
    libzbar0\
    libzbar-dev &&\
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /usr/src/weapon-detection

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt /usr/src/weapon-detection/
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
RUN pip install --no-cache ultralytics --no-deps

# Copy the current directory contents into the container at /usr/src/weapon-detection
COPY . /usr/src/weapon-detection

# Set environment variables
ENV OMP_NUM_THREADS=1

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
#CMD ["python3", "app.py"]