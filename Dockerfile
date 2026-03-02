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

# Install only what PSHuman inference actually needs (skip full requirements.txt
# which has 128 packages including conflicting torch/CUDA/TensorRT versions)
WORKDIR $PSHUMAN_DIR
RUN pip install --no-cache-dir \
    diffusers transformers huggingface_hub accelerate safetensors \
    omegaconf einops configargparse \
    opencv-python-headless Pillow scikit-image imageio \
    kornia open3d trimesh plyfile \
    rembg[gpu] pymatting \
    tqdm peft

# RunPod SDK + brotli (required for RunPod job fetching)
# v3 cache bust - force fresh install
RUN pip install --no-cache-dir runpod brotli && \
    python -c "import brotli; print('brotli OK')" && \
    python -c "import runpod; print(f'runpod {runpod.__version__}')"

# NOTE: Model weights (~5GB) are downloaded on first cold start to keep image small.
# This adds ~2min to the first request only.

# Copy the handler
COPY handler.py /workspace/handler.py

# RunPod serverless entry point
WORKDIR /workspace
CMD ["python", "handler.py"]
