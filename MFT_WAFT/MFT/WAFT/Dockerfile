# Use NVIDIA PyTorch NGC container
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Update package lists
RUN apt-get update

# Set working directory
WORKDIR /usr/src/app

# Suppress interactive prompts
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install sudo dialog apt-utils

# Install useful tools and OpenCV dependencies
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get install -y -q gedit libglib2.0-0 libsm6 libxrender1 libxext6

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install them
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install \
    -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp

# Copy your project files into the image
COPY . /usr/src/app

# Default command to run your code
CMD ["python3", "main.py"]
