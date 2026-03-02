# PSHuman RunPod Serverless Deployment

Run PSHuman 3D human reconstruction on cloud GPUs (A100/A6000) via RunPod Serverless.
Pay only for actual inference time (~60s per image at ~$0.0004/sec on A100).

## Prerequisites

1. **RunPod account**: Sign up at https://runpod.io
2. **Docker Hub account**: For hosting the container image
3. **SMPL-X model files**: Download from https://smpl-x.is.tue.mpg.de/ (academic license)

## Deployment Steps

### 1. Build the Docker image

```bash
cd runpod_pshuman

# If you have SMPL-X models, place them here first:
# mkdir -p smplx_models && cp -r /path/to/smplx_models/* smplx_models/
# Then uncomment the SMPL-X lines in the Dockerfile.

docker build -t yourdockerhub/pshuman-runpod:latest .
docker push yourdockerhub/pshuman-runpod:latest
```

**Note**: The image is large (~15-20GB) due to PyTorch + model weights baked in.
This eliminates cold-start model downloads.

### 2. Create RunPod Serverless Endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click **New Endpoint**
3. Settings:
   - **Container Image**: `yourdockerhub/pshuman-runpod:latest`
   - **GPU Type**: A100 (80GB) or A6000 (48GB)
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 1 (or more for parallel requests)
   - **Idle Timeout**: 30s (keep warm briefly after job)
   - **Execution Timeout**: 600s
   - **Container Disk**: 30GB
4. Click **Create**
5. Note the **Endpoint ID** from the dashboard

### 3. Get RunPod API Key

1. Go to https://www.runpod.io/console/user/settings
2. Under **API Keys**, create a new key
3. Copy the key

### 4. Configure Persona 3.0 Backend

Set environment variables before starting the backend:

```bash
# Windows (PowerShell)
$env:RUNPOD_API_KEY = "your_runpod_api_key_here"
$env:RUNPOD_ENDPOINT_ID = "your_endpoint_id_here"

# Windows (CMD)
set RUNPOD_API_KEY=your_runpod_api_key_here
set RUNPOD_ENDPOINT_ID=your_endpoint_id_here

# Linux/Mac
export RUNPOD_API_KEY=your_runpod_api_key_here
export RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

Or add them to a `.env` file in the backend directory.

Then start the backend as usual:
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Test

Check endpoint health:
```bash
curl http://localhost:8000/api/body/pshuman_status
```

Should return:
```json
{
  "configured": true,
  "endpoint_id": "abc12345...",
  "healthy": true,
  "workers": {"idle": 0, "running": 0, "throttled": 0}
}
```

## Usage

In the Persona 3.0 frontend:
1. Upload a body photo
2. Click **"PSHuman 3D (Cloud GPU)"**
3. Wait ~60-90s (first call may take 2-3 min for cold start)
4. The reconstructed mesh appears as the PSHuman layer

## Cost Estimate

- **A100 80GB**: ~$0.0012/sec → ~$0.07 per reconstruction (~60s)
- **A6000 48GB**: ~$0.0006/sec → ~$0.04 per reconstruction (~60s)
- **Cold start**: First request after idle adds ~30-120s (GPU allocation)
- **Idle cost**: $0 (scales to zero)

## Architecture

```
Frontend (browser)
    ↓ photo upload
Local Backend (localhost:8000)
    ↓ POST /api/body/pshuman_reconstruct
    ↓ sends base64 image to RunPod
RunPod Serverless (A100/A6000)
    ↓ PSHuman inference (~60s)
    ↓ returns vertex-colored mesh
Local Backend
    ↓ SMPL-X skeleton rigging (local, ~2s)
    ↓ returns rigged mesh
Frontend
    ↓ renders PSHumanMesh component
```

## Files

- `handler.py` — RunPod serverless handler (receives image, runs PSHuman, returns mesh)
- `Dockerfile` — Container build with PyTorch, Kaolin, PSHuman, model weights
- `README.md` — This file

## Troubleshooting

**"PSHuman cloud not configured"**: Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID env vars.

**Cold start timeout**: First request may take 2-3 min. Increase the frontend timeout or use RunPod's async `/run` + `/status` polling.

**SMPL-X models missing**: Download from https://smpl-x.is.tue.mpg.de/ and bake into Docker image.

**Out of VRAM**: PSHuman needs 40GB+ at 768 resolution. Use A100 80GB or A6000 48GB.
