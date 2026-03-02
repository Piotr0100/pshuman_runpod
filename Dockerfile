# PSHuman RunPod Serverless Dockerfile
# GPU: A100 (40/80GB) or A6000 (48GB)
# Uses PyTorch base image to avoid torch installation issues.

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PSHUMAN_DIR=/workspace/PSHuman

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 \
    libxext6 libegl1-mesa-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Kaolin (NVIDIA 3D toolkit, needed by PSHuman)
RUN pip install --no-cache-dir kaolin==0.17.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html

# Clone PSHuman
RUN git clone https://github.com/pengHTYX/PSHuman.git $PSHUMAN_DIR

# Install PSHuman requirements
WORKDIR $PSHUMAN_DIR
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod SDK + utility packages
RUN pip install --no-cache-dir runpod trimesh plyfile "rembg[gpu]"

# Pre-download the HuggingFace model weights at build time
# This avoids downloading on first request (saves ~2min cold start)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('pengHTYX/PSHuman_Unclip_768_6views', local_dir='/workspace/models/PSHuman_Unclip_768_6views')\
"

# Copy the handler
COPY handler.py /workspace/handler.py

# RunPod serverless entry point
WORKDIR /workspace
CMD ["python", "handler.py"]
