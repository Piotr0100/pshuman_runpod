# PSHuman RunPod Serverless Dockerfile
# GPU: A100 (40/80GB) or A6000 (48GB)
# Build: docker build -t pshuman-runpod .
# Push to Docker Hub or RunPod registry, then create serverless endpoint.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PSHUMAN_DIR=/workspace/PSHuman

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git wget curl libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 \
    libxext6 libegl1-mesa-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.1
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Kaolin (NVIDIA 3D toolkit, needed by PSHuman)
RUN pip install kaolin==0.17.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html

# Clone PSHuman
RUN git clone https://github.com/pengHTYX/PSHuman.git $PSHUMAN_DIR

# Install PSHuman requirements
WORKDIR $PSHUMAN_DIR
RUN pip install -r requirements.txt

# Install RunPod SDK + utility packages
RUN pip install runpod trimesh plyfile rembg[gpu]

# Pre-download the HuggingFace model weights at build time
# This avoids downloading on first request (saves ~2min cold start)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('pengHTYX/PSHuman_Unclip_768_6views', local_dir='/workspace/models/PSHuman_Unclip_768_6views')\
"

# Download SMPL-X models if needed by PSHuman
# NOTE: You may need to manually download SMPL-X from https://smpl-x.is.tue.mpg.de/
# and place them in $PSHUMAN_DIR/smplx_models/ before building.
# Uncomment if you have a download URL:
# RUN mkdir -p $PSHUMAN_DIR/smplx_models && \
#     wget -O /tmp/smplx.zip "YOUR_SMPLX_DOWNLOAD_URL" && \
#     unzip /tmp/smplx.zip -d $PSHUMAN_DIR/smplx_models/ && \
#     rm /tmp/smplx.zip

# Copy the handler
COPY handler.py /workspace/handler.py

# RunPod serverless entry point
WORKDIR /workspace
CMD ["python", "handler.py"]
