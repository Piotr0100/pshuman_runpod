"""
RunPod Serverless Handler for PSHuman.

Receives a single RGBA image (base64-encoded PNG), runs the full PSHuman
pipeline (multiview diffusion → reconstruction → remeshing), and returns
the textured mesh as a vertex-colored PLY (base64-encoded).

Environment:
  - PSHuman repo cloned at /workspace/PSHuman
  - Model weights cached at /workspace/models
  - CUDA GPU with ≥40GB VRAM (A100 or A6000)

Input JSON:
  {
    "input": {
      "image_b64": "<base64 PNG, background already removed>",
      "crop_size": 740,       # optional, default 740
      "seed": 600,            # optional, default 600
      "with_smpl": false      # optional, default false
    }
  }

Output JSON:
  {
    "mesh_vertices_b64": "<base64 float32, flat (N*3)>",
    "mesh_faces_b64": "<base64 int32, flat (F*3)>",
    "mesh_colors_b64": "<base64 float32, flat (N*3), RGB in [0,1]>",
    "n_vertices": N,
    "n_faces": F,
    "processing_time": seconds
  }
"""

import runpod
import base64
import io
import os
import sys
import time
import tempfile
import json
import numpy as np

# Add PSHuman to path
PSHUMAN_DIR = os.environ.get("PSHUMAN_DIR", "/workspace/PSHuman")
sys.path.insert(0, PSHUMAN_DIR)


def encode_array(arr):
    """Encode numpy array as base64 string."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def encode_int_array(arr):
    """Encode numpy int array as base64 string."""
    return base64.b64encode(arr.astype(np.int32).tobytes()).decode("ascii")


def load_mesh_ply(ply_path):
    """Load a PLY mesh with vertex colors, return vertices, faces, colors."""
    import trimesh

    mesh = trimesh.load(ply_path, process=False)

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Extract vertex colors (RGBA uint8 → RGB float [0,1])
    if mesh.visual and hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    else:
        # Fallback: neutral skin tone
        colors = np.full((vertices.shape[0], 3), [0.82, 0.72, 0.62], dtype=np.float32)

    return vertices, faces, colors


def load_mesh_obj(obj_path):
    """Load an OBJ mesh with vertex colors, return vertices, faces, colors."""
    import trimesh

    mesh = trimesh.load(obj_path, process=False)

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    if mesh.visual and hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    else:
        colors = np.full((vertices.shape[0], 3), [0.82, 0.72, 0.62], dtype=np.float32)

    return vertices, faces, colors


def handler(job):
    """RunPod serverless handler — PSHuman single-image reconstruction."""
    t0 = time.time()
    job_input = job["input"]

    # ── Parse input ──
    image_b64 = job_input.get("image_b64")
    if not image_b64:
        return {"error": "Missing 'image_b64' in input"}

    crop_size = job_input.get("crop_size", 740)
    seed = job_input.get("seed", 600)
    with_smpl = job_input.get("with_smpl", False)

    # ── Save input image to temp file ──
    image_bytes = base64.b64decode(image_b64)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # Write RGBA PNG
        input_path = os.path.join(input_dir, "input.png")
        with open(input_path, "wb") as f:
            f.write(image_bytes)

        print(f"[PSHuman] Input image saved: {len(image_bytes)} bytes")
        print(f"[PSHuman] Settings: crop_size={crop_size}, seed={seed}, with_smpl={with_smpl}")

        # ── Run PSHuman inference ──
        # We shell out to inference.py to keep the pipeline self-contained
        # and avoid import conflicts with the handler's environment.
        import subprocess

        cmd = [
            sys.executable, os.path.join(PSHUMAN_DIR, "inference.py"),
            "--config", os.path.join(PSHUMAN_DIR, "configs/inference-768-6view.yaml"),
            f"pretrained_model_name_or_path=pengHTYX/PSHuman_Unclip_768_6views",
            f"validation_dataset.crop_size={crop_size}",
            f"with_smpl={'true' if with_smpl else 'false'}",
            f"validation_dataset.root_dir={input_dir}",
            f"seed={seed}",
            "num_views=7",
            "save_mode=rgb",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # PSHuman saves output to ./out/ by default — override via cwd or config
        # We'll run from a temp directory and look for output there
        pshuman_out = os.path.join(PSHUMAN_DIR, "out")

        print(f"[PSHuman] Running inference...")
        result = subprocess.run(
            cmd,
            cwd=PSHUMAN_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"[PSHuman] STDERR: {result.stderr[-2000:]}")
            return {
                "error": f"PSHuman inference failed (exit code {result.returncode})",
                "stderr": result.stderr[-2000:],
            }

        print(f"[PSHuman] Inference completed in {time.time() - t0:.1f}s")
        print(f"[PSHuman] STDOUT (last 500): {result.stdout[-500:]}")

        # ── Find output mesh ──
        # PSHuman saves to out/<case_name>/ — look for final OBJ or PLY
        mesh_path = None
        for root, dirs, files in os.walk(pshuman_out):
            for f in files:
                if f.endswith("_final.obj") or f.endswith("_remeshed.obj"):
                    mesh_path = os.path.join(root, f)
                    break
                if f.endswith(".obj") and "result_clr" in f:
                    mesh_path = os.path.join(root, f)
                    break
                if f.endswith(".ply") and mesh_path is None:
                    mesh_path = os.path.join(root, f)
            if mesh_path:
                break

        if not mesh_path or not os.path.exists(mesh_path):
            # List what's in the output dir for debugging
            found_files = []
            if os.path.exists(pshuman_out):
                for root, dirs, files in os.walk(pshuman_out):
                    for f in files:
                        found_files.append(os.path.join(root, f))
            return {
                "error": "No output mesh found after PSHuman inference",
                "output_files": found_files[-20:],
            }

        print(f"[PSHuman] Loading output mesh: {mesh_path}")

        # ── Load mesh ──
        if mesh_path.endswith(".ply"):
            vertices, faces, colors = load_mesh_ply(mesh_path)
        else:
            vertices, faces, colors = load_mesh_obj(mesh_path)

        n_verts = vertices.shape[0]
        n_faces = faces.shape[0]
        elapsed = time.time() - t0

        print(f"[PSHuman] Output: {n_verts} verts, {n_faces} faces, {elapsed:.1f}s total")

        # ── Clean up PSHuman output directory ──
        import shutil
        try:
            # Clean the specific case output, not the whole out/ dir
            for item in os.listdir(pshuman_out):
                item_path = os.path.join(pshuman_out, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        except Exception as e:
            print(f"[PSHuman] Cleanup warning: {e}")

        # ── Return result ──
        return {
            "mesh_vertices_b64": encode_array(vertices.flatten()),
            "mesh_faces_b64": encode_int_array(faces.flatten()),
            "mesh_colors_b64": encode_array(colors.flatten()),
            "n_vertices": n_verts,
            "n_faces": n_faces,
            "processing_time": round(elapsed, 2),
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
