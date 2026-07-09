---
title: flux_trajectoid
emoji: 🪨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# flux_trajectoid · Photon Seed Asteroids

Single-page visual demo of **trajectoid shells + VQC/OAM + oam_flux lattice**.

- **Space:** https://huggingface.co/spaces/kinaar111/flux_trajectoid  
- **Code:** https://github.com/kinaar8340/flux_trajectoid  

## Features

- 25/50/25 layout · CONTROLS | REFERENCES · 2×2 viewports  
- Matrix slice planes **x / y / z / xyz** with synced shell · radial · path  
- Green scan frame + intersection arc · plane-specific trench heatmaps  
- Play matrix scan GIFs (XYZ sequences X→Y→Z)  

## Local run

From the repo root:

```bash
cd ~/Projects/flux_trajectoid
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r space/requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python space/app.py
# → http://127.0.0.1:7860
```

## Deploy

Space root is the contents of `space/` (including vendored `flux_trajectoid/`).
