---
title: flux_trajectoid
emoji: 🪨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# flux_trajectoid · Photon Seed Asteroids

Single-page visual demo of **trajectoid shells + VQC/OAM + oam_flux lattice**.

- **Space:** https://huggingface.co/spaces/kinaar111/flux_trajectoid  
- **Code:** https://github.com/kinaar8340/flux_trajectoid  

## Local run (keep local until ready to push)

From the repo root:

```bash
cd ~/Projects/flux_trajectoid
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r space/requirements.txt

# Ensure package import works from space/
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python space/app.py
# → http://127.0.0.1:7860
```

Or:

```bash
cd space
PYTHONPATH=../src python app.py
```

## UI notes

- Fixed viewport layout (no page scroll); top navigation bar  
- Multi-column viewports: shell · radial · field · path · metrics  
- Glass layers use **opacity 0.3** on inner/foreground panels  
- Hero visual: Nature-style trajectoid gallery (`assets/trajectoid_paths.png`)  

## HF deploy (later)

Copy `space/` contents to the Space repo root (or set Space to use `space/` subdirectory), ensure `requirements.txt` installs `flux-trajectoid` from git or vendor `src/`.
