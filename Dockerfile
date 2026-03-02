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

# v8 - Install PSHuman's full requirements.txt but pin critical versions first
# and filter out packages that conflict with our base image
WORKDIR $PSHUMAN_DIR

# Pin versions that must be compatible with PyTorch 2.1
RUN pip install --no-cache-dir \
    "numpy<2" \
    "diffusers==0.27.2" "transformers==4.40.2" "huggingface_hub==0.23.5" \
    "accelerate==0.29.3" safetensors

# Install everything from requirements.txt, skipping torch/cuda/tensorrt
# (already in base image) and ignoring version conflicts
RUN grep -viE "^(torch|nvidia|tensorrt|triton|xformers|pytorch3d|nvdiffrast|jax)" requirements.txt \
    | pip install --no-cache-dir --ignore-installed -r /dev/stdin \
    || echo "Some packages failed — continuing anyway"

# Re-pin our critical versions that requirements.txt may have overwritten
RUN pip install --no-cache-dir --force-reinstall \
    "numpy<2" \
    "diffusers==0.27.2" "transformers==4.40.2" "huggingface_hub==0.23.5" \
    "accelerate==0.29.3"

# RunPod SDK with brotli support for aiohttp content decoding
# v4 - install all brotli variants + upgrade aiohttp + latest runpod
RUN pip install --no-cache-dir --upgrade \
    runpod \
    aiohttp[speedups] \
    brotli \
    brotlicffi \
    && python -c "import brotli; print('brotli OK')" \
    && python -c "import aiohttp; print(f'aiohttp {aiohttp.__version__}')" \
    && python -c "import runpod; print(f'runpod {runpod.__version__}')"

# NOTE: Model weights (~5GB) are downloaded on first cold start to keep image small.
# This adds ~2min to the first request only.

# Copy the handler
COPY handler.py /workspace/handler.py

# RunPod serverless entry point
WORKDIR /workspace
CMD ["python", "handler.py"]
