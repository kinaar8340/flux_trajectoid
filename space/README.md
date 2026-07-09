---
title: flux_trajectoid
emoji: 🪨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.23.2
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

- Full-viewport layout · CONTROLS | REFERENCES · **2×2 viewports**
  (shell · rolling path · radial trench · turbulence scorecard)
- Matrix slice planes **x / y / z / xyz** with synced shell · radial · path  
- Green scan frame + intersection arc · plane-specific trench heatmaps  
- Play matrix scan GIFs (XYZ sequences X→Y→Z)  
- **SLM export** — download phase hologram zip (device presets)  
- Scorecard legend: **P / Strehl / OAMf / Icorr / F**  

**Demo video:** https://x.com/kinaar111/status/2075134240703029650

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
