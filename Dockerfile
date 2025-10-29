# Base image with PyTorch
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# Set working directory inside container
WORKDIR /workspace

# Copy everything from your submission folder into the image
COPY . /workspace/agardiner_STIR_submission

# === Install MFT_WAFT ===
WORKDIR /workspace/agardiner_STIR_submission/MFT_WAFT
RUN pip install .

# === Install STIRLoader ===
WORKDIR /workspace/agardiner_STIR_submission/STIRLoader-main
RUN pip install .

# === Extra dependencies (from shared Dockerfile) ===
RUN pip install torchvision onnxruntime-gpu timm
RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6

# === Set final working directory for runtime ===
WORKDIR /workspace/agardiner_STIR_submission/STIRMetrics-main/src
