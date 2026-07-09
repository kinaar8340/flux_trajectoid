#!/usr/bin/env python3
"""
flux_trajectoid — single-page Gradio Space (local-first).

HF Space: https://huggingface.co/spaces/kinaar111/flux_trajectoid
GitHub:   https://github.com/kinaar8340/flux_trajectoid

Layout: full-viewport page, top nav, multi-column viewports.
Inner / foreground layers use opacity 0.3 glass panels.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr
import numpy as np

from demo_core import (
    ASSETS,
    GITHUB,
    HF_SPACE,
    animate_matrix_scan,
    asset_path,
    blank_rgb,
    replot_shell_only,
    run_pipeline,
)

# Keep import alias clear for suite return

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme / CSS — fixed single page, glass layers @ 0.3 opacity
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
html, body, .gradio-container {
  height: 100% !important;
  max-height: 100vh !important;
  overflow: hidden !important;
  margin: 0 !important;
  background: #070b14 !important;
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif !important;
}
.gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
}
/* Kill default scroll on main fill */
.main, .wrap, .contain {
  max-height: 100vh !important;
  overflow: hidden !important;
}
footer { display: none !important; }

/*
 * theme_default_tabs
 *   idle_text   = #64748b
 *   active_text = #00FF00  (tab_state=True)
 *   active_bar  = 2px #00FF00 underline
 *   rail        = thin dim baseline under tab row
 * Applied: main #nav-bar · col-1 #col1-tabs
 */
:root {
  --ft-tab-idle: #64748b;
  --ft-tab-hover: #94a3b8;
  --ft-tab-active: #00FF00;
  --ft-tab-rail: rgba(148, 163, 184, 0.22);
  --ft-tab-font: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
}

/* Top navigation — same tab language as CONTROLS | REFERENCES */
#nav-bar {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  padding: 0.35rem 0.75rem 0;
  background: rgba(15, 23, 42, 0.92);
  border-bottom: none;
  min-height: 44px;
  max-height: 52px;
  position: relative;
}
#nav-bar::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 1px;
  background: var(--ft-tab-rail);
  pointer-events: none;
}
#nav-bar .nav-title {
  color: #e2e8f0;
  font-weight: 600;
  font-size: 0.95rem;
  margin-right: 0.5rem;
  letter-spacing: 0.02em;
  padding-bottom: 0.45rem;
}
#nav-bar button,
#nav-bar button span {
  position: relative !important;
  color: var(--ft-tab-idle) !important;
  font-family: var(--ft-tab-font) !important;
  font-size: 0.78rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.04em !important;
  padding: 0.35rem 0.2rem 0.55rem 0.2rem !important;
  border: none !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  min-width: auto !important;
  height: auto !important;
}
#nav-bar button:hover,
#nav-bar button:hover span {
  color: var(--ft-tab-hover) !important;
  background: transparent !important;
  border: none !important;
}
/* active main nav tab */
#nav-bar button.nav-active,
#nav-bar button.nav-active span,
#nav-bar button.primary,
#nav-bar button.primary span,
#nav-bar button.selected,
#nav-bar button.selected span {
  color: var(--ft-tab-active) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#nav-bar button.nav-active::after,
#nav-bar button.primary::after,
#nav-bar button.selected::after {
  content: "" !important;
  position: absolute !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  height: 2px !important;
  background: var(--ft-tab-active) !important;
  border-radius: 1px !important;
  z-index: 1 !important;
}
#nav-meta {
  margin-left: auto;
  color: #64748b;
  font-size: 0.72rem;
  padding-bottom: 0.45rem;
}
#nav-meta a {
  color: #64748b !important;
  text-decoration: none !important;
}
#nav-meta a:hover {
  color: var(--ft-tab-hover) !important;
}

/* Glass / translucent layers — user requested opacity 0.3 */
.layer-inner, .layer-fg {
  background: rgba(15, 23, 42, 0.3) !important;
  border: 1px solid rgba(148, 163, 184, 0.18) !important;
  border-radius: 10px !important;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}
.layer-panel {
  background: rgba(30, 41, 59, 0.3) !important;
  border: 1px solid rgba(56, 189, 248, 0.15) !important;
  border-radius: 8px !important;
  padding: 0.35rem 0.5rem !important;
}

/* Main workspace fits under nav */
#workspace {
  height: calc(100vh - 52px) !important;
  max-height: calc(100vh - 52px) !important;
  overflow: hidden !important;
  padding: 0.4rem 0.55rem !important;
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(59, 130, 246, 0.12), transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 100%, rgba(167, 139, 250, 0.10), transparent 50%),
    #070b14 !important;
}
/* Direct children of workspace row stretch full height */
#workspace > div {
  height: 100% !important;
  min-height: 0 !important;
  max-height: 100% !important;
}

/* Col-1 scrolls independently; viewports use fixed balanced grid */
#controls {
  height: 100% !important;
  max-height: 100% !important;
  min-height: 0 !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  scrollbar-width: thin;
  scrollbar-color: rgba(56, 189, 248, 0.45) rgba(15, 23, 42, 0.4);
}
#controls::-webkit-scrollbar {
  width: 8px;
}
#controls::-webkit-scrollbar-thumb {
  background: rgba(56, 189, 248, 0.4);
  border-radius: 4px;
}

/*
 * Balanced viewport grid (cols 2–3 merged)
 *   outer shell: #viewport-grid (no glass — just layout)
 *   glass cells:  .vp-cell.layer-inner (plot frame)
 *   plot layer:   Gradio Image inside cell (must keep min-height)
 *   row heights equal → shared horizontals; col scale 4:3
 */
#viewport-grid {
  height: 100% !important;
  max-height: 100% !important;
  min-height: 0 !important;
  overflow: hidden !important;
  padding: 0 !important;
  /* layout only — do not paint a glass layer here */
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
}
/* Gradio wraps Column children; force the wrap + rows to fill */
#viewport-grid,
#viewport-grid > .wrap,
#viewport-grid > .form,
#viewport-grid > div {
  display: flex !important;
  flex-direction: column !important;
  height: 100% !important;
  min-height: 0 !important;
  gap: 0.35rem !important;
}
/* Equal row bands (fixed horizontal baselines across both columns) */
#viewport-grid .vp-row {
  flex: 1 1 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
  gap: 0.35rem !important;
  align-items: stretch !important;
  display: flex !important;
  flex-direction: row !important;
}
/* If Gradio nests the row, still stretch its direct children */
#viewport-grid .vp-row > div,
#viewport-grid .vp-row > .form,
#viewport-grid .vp-row > .wrap {
  display: flex !important;
  flex-direction: row !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: 100% !important;
  gap: 0.35rem !important;
  align-items: stretch !important;
}
/* Glass frame = outer cell only (not the image itself) */
#viewport-grid .vp-cell {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 0 !important;
  min-height: 0 !important;
  height: 100% !important;
  overflow: hidden !important;
  padding: 0.35rem 0.45rem !important;
  /* layer-inner supplies glass; ensure it doesn't clip plots to 0 */
  box-sizing: border-box !important;
}
#viewport-grid .vp-cell .viewport-title {
  flex: 0 0 auto !important;
  margin: 0 0 0.25rem 0 !important;
}
/* Gradio Image block = inner plot layer; keep real height */
#viewport-grid .vp-cell .block,
#viewport-grid .vp-cell [data-testid="image"] {
  flex: 1 1 auto !important;
  min-height: 140px !important;
  height: auto !important;
  max-height: 100% !important;
  overflow: hidden !important;
  background: rgba(7, 11, 20, 0.55) !important;
  border: 1px solid rgba(148, 163, 184, 0.12) !important;
  border-radius: 8px !important;
}
#viewport-grid .vp-cell .image-container {
  width: 100% !important;
  height: 100% !important;
  min-height: 140px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: hidden !important;
}
#viewport-grid .vp-cell img {
  width: 100% !important;
  height: auto !important;
  max-height: 100% !important;
  object-fit: contain !important;
  display: block !important;
  opacity: 1 !important;
}

/* Column 1: compact top stack + Seed Status fills remainder */
#controls {
  display: flex !important;
  flex-direction: column !important;
  padding: 0.35rem 0.4rem !important;
  gap: 0.25rem !important;
}
#controls-top {
  flex: 0 0 auto !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 0.2rem !important;
  min-height: 0 !important;
}
#controls .viewport-title {
  margin: 0 0 0.1rem 0 !important;
  font-size: 0.65rem !important;
  line-height: 1.1 !important;
}
/* Column-1: CONTROLS | REFERENCES — theme_default_tabs
 * Gradio 6: .tab-wrapper > .tab-container > button.selected
 */
#col1-tabs {
  --color-accent: var(--ft-tab-active);
  margin: 0 !important;
  padding: 0 !important;
  gap: 0 !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
}
#col1-tabs .tab-wrapper,
#controls #col1-tabs .tab-wrapper {
  height: auto !important;
  min-height: 1.5rem !important;
  padding-bottom: 0.15rem !important;
  margin-bottom: 0.35rem !important;
  background: transparent !important;
}
#col1-tabs .tab-container,
#controls #col1-tabs .tab-container {
  height: auto !important;
  min-height: 1.4rem !important;
  gap: 0.65rem !important;
  overflow: visible !important;
  background: transparent !important;
}
#col1-tabs .tab-container:after {
  background-color: rgba(148, 163, 184, 0.18) !important;
  height: 1px !important;
}
/* Idle tabs — match .viewport-title / CONTROLS label type */
#col1-tabs .tab-container button,
#col1-tabs .tab-wrapper button,
#controls #col1-tabs .tab-container button,
#controls #col1-tabs .tab-wrapper button,
#controls #col1-tabs .tab-container button span,
#controls #col1-tabs .tab-wrapper button span {
  color: var(--ft-tab-idle) !important;
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif !important;
  font-size: 0.7rem !important;
  font-weight: 400 !important;
  line-height: 1.1 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  padding: 0.15rem 0.1rem 0.4rem 0.1rem !important;
  margin: 0 !important;
  height: auto !important;
  min-width: auto !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  flex: 0 0 auto !important;
}
#col1-tabs .tab-container button:hover:not(.selected),
#controls #col1-tabs .tab-container button:hover:not(.selected) {
  color: #94a3b8 !important;
  background: transparent !important;
}
/* active_tab (tab_state=True): CONTROLS or REFERENCES → #00FF00 */
#col1-tabs .tab-container button.selected,
#col1-tabs .tab-wrapper button.selected,
#col1-tabs button.selected,
#controls #col1-tabs .tab-container button.selected,
#controls #col1-tabs .tab-wrapper button.selected,
#controls #col1-tabs button.selected,
#controls #col1-tabs .tab-container button.selected span,
#controls #col1-tabs .tab-wrapper button.selected span,
#controls #col1-tabs button.selected span {
  color: var(--ft-tab-active) !important;
  background: transparent !important;
  font-weight: 400 !important;
}
/* Gradio draws active underline via button.selected::after */
#col1-tabs .tab-container button.selected:after,
#col1-tabs button.selected:after,
#controls #col1-tabs button.selected:after {
  background-color: var(--ft-tab-active) !important;
  height: 2px !important;
}
#col1-tabs .tabitem,
#col1-tabs [role="tabpanel"] {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: auto !important;
  padding: 0 !important;
  border: none !important;
}
#panel-refs {
  display: flex !important;
  flex-direction: column !important;
  gap: 0.35rem !important;
  min-height: 0 !important;
}
#panel-refs img,
#panel-refs .image-container {
  width: 100% !important;
  max-height: none !important;
  object-fit: contain !important;
}
#controls label, #controls span, #controls p {
  color: #94a3b8 !important;
  font-size: 0.68rem !important;
}
#controls input, #controls textarea {
  background: rgba(15, 23, 42, 0.55) !important;
  border-color: rgba(100, 116, 139, 0.4) !important;
  color: #e2e8f0 !important;
  font-size: 0.75rem !important;
  min-height: 1.6rem !important;
  padding: 0.15rem 0.35rem !important;
}
/*
 * theme_default_slider
 *   analog_fill_color     = #00FF00  (left of knob only)
 *   analog_bar_height     = ~1.5px
 *   analog_effect_glowing = True (thumb + fill layer)
 *
 * Fill is a real DOM inner layer (.ft-slider-fill) whose width tracks
 * the knob — pseudo-element track gradients were painting full-bar green.
 */
:root {
  --ft-slider-fill: #00FF00;
  --ft-slider-rail: rgba(100, 116, 139, 0.45);
  --ft-slider-h: 1.5px;
  --ft-slider-thumb: 10px;
  --ft-slider-glow: 0 0 4px 1px rgba(0, 255, 0, 0.55),
                    0 0 8px 2px rgba(0, 255, 0, 0.28);
}
/* Shell: rail + fill sit under a transparent range input */
.ft-slider-shell {
  position: relative !important;
  flex: 1 1 auto !important;
  width: 100% !important;
  min-width: 0 !important;
  height: var(--ft-slider-thumb) !important;
  display: flex !important;
  align-items: center !important;
  margin: 0.2rem 0 0.4rem 0 !important;
}
/* Dim full-width rail (always visible baseline) */
.ft-slider-rail {
  position: absolute !important;
  left: 0 !important;
  right: 0 !important;
  top: 50% !important;
  height: var(--ft-slider-h) !important;
  transform: translateY(-50%) !important;
  border-radius: 999px !important;
  background: var(--ft-slider-rail) !important;
  pointer-events: none !important;
  z-index: 0 !important;
  overflow: visible !important; /* allow green glow bloom on fill */
}
/* Inner layer — green glowing line from 0 → knob (width set by JS) */
.ft-slider-fill {
  position: absolute !important;
  left: 0 !important;
  top: 0 !important;
  bottom: 0 !important;
  width: 0%;
  min-width: 0;
  border-radius: 999px !important;
  background: var(--ft-slider-fill) !important;
  box-shadow:
    0 0 3px 0.5px rgba(0, 255, 0, 0.95),
    0 0 6px 1px rgba(0, 255, 0, 0.55),
    0 0 12px 2px rgba(0, 255, 0, 0.28) !important;
  pointer-events: none !important;
  z-index: 1 !important;
  transition: none !important;
}
/* Range sits on top; its own track is fully invisible */
.ft-slider-shell > input[type="range"],
#controls input[type="range"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  position: relative !important;
  z-index: 2 !important;
  width: 100% !important;
  height: var(--ft-slider-thumb) !important;
  min-height: var(--ft-slider-thumb) !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  outline: none !important;
  cursor: pointer !important;
  background: transparent !important;
  box-shadow: none !important;
  accent-color: transparent !important;
  --slider-color: transparent !important;
  --range_progress: 0% !important;
}
.ft-slider-shell > input[type="range"]::-webkit-slider-runnable-track,
#controls input[type="range"]::-webkit-slider-runnable-track {
  height: var(--ft-slider-h) !important;
  border-radius: 999px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.ft-slider-shell > input[type="range"]::-webkit-slider-thumb,
#controls input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none !important;
  appearance: none !important;
  width: var(--ft-slider-thumb) !important;
  height: var(--ft-slider-thumb) !important;
  margin-top: calc((var(--ft-slider-h) - var(--ft-slider-thumb)) / 2) !important;
  border-radius: 50% !important;
  background: #f8fafc !important;
  border: 1.5px solid var(--ft-slider-fill) !important;
  box-shadow: var(--ft-slider-glow) !important;
  cursor: pointer !important;
  position: relative !important;
  z-index: 3 !important;
}
.ft-slider-shell > input[type="range"]::-moz-range-track,
#controls input[type="range"]::-moz-range-track {
  height: var(--ft-slider-h) !important;
  border-radius: 999px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.ft-slider-shell > input[type="range"]::-moz-range-progress,
#controls input[type="range"]::-moz-range-progress {
  height: var(--ft-slider-h) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.ft-slider-shell > input[type="range"]::-moz-range-thumb,
#controls input[type="range"]::-moz-range-thumb {
  width: var(--ft-slider-thumb) !important;
  height: var(--ft-slider-thumb) !important;
  border-radius: 50% !important;
  background: #f8fafc !important;
  border: 1.5px solid var(--ft-slider-fill) !important;
  box-shadow: var(--ft-slider-glow) !important;
  cursor: pointer !important;
}
/* Collapsed accordion headers ~ one compact row each */
#controls .label-wrap,
#controls .accordion,
#controls details {
  background: rgba(15, 23, 42, 0.3) !important;
  border: 1px solid rgba(148, 163, 184, 0.15) !important;
  border-radius: 6px !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 0 !important;
}
#controls .label-wrap {
  padding: 0.2rem 0.45rem !important;
  min-height: 1.55rem !important;
  max-height: 1.7rem !important;
}
#controls summary,
#controls .label-wrap span {
  color: #cbd5e1 !important;
  font-size: 0.7rem !important;
  line-height: 1.2 !important;
  padding: 0 !important;
  margin: 0 !important;
}
#controls .gradio-accordion,
#controls [class*="accordion"] {
  margin: 0 !important;
  padding: 0 !important;
}
#run-btn {
  background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
  color: white !important;
  border: none !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  min-height: 1.85rem !important;
  max-height: 2rem !important;
  padding: 0.25rem 0.5rem !important;
  margin: 0.15rem 0 0 0 !important;
}
#scan-btn {
  background: rgba(0, 255, 0, 0.12) !important;
  color: #86efac !important;
  border: 1px solid rgba(0, 255, 0, 0.45) !important;
  font-size: 0.72rem !important;
  min-height: 1.6rem !important;
  margin-top: 0.15rem !important;
}

/*
 * Global selection theme defaults:
 *   default_button_style = circle + concentric border ring
 *   button_body_height   = 1 row
 *   button_effect_glowing = True when latched (button_state=True)
 * Latched: ONLY the concentric ring / border → #00FF00 + mild glow
 * (inner disk stays dark; outer chip stays neutral)
 * Latched until another radio in the group is selected.
 */
:root {
  --ft-circle-size: 1.05em;      /* outer circle diameter (1-row) */
  --ft-ring-width: 2px;          /* concentric border thickness */
  --ft-btn-row-h: 1.45rem;
  --ft-ring-idle: rgba(148, 163, 184, 0.65);
  --ft-ring-active: #00FF00;
  --ft-disk: rgba(15, 23, 42, 0.95);
  --ft-glow: 0 0 5px 1px rgba(0, 255, 0, 0.5),
             0 0 11px 2px rgba(0, 255, 0, 0.25);
}
#controls input[type="checkbox"],
#controls input[type="radio"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  width: var(--ft-circle-size) !important;
  height: var(--ft-circle-size) !important;
  border-radius: 50% !important;                 /* circle */
  border: var(--ft-ring-width) solid var(--ft-ring-idle) !important;  /* concentric ring */
  background: var(--ft-disk) !important;         /* inner disk — never green fill */
  background-image: none !important;
  box-shadow: none !important;
  flex-shrink: 0 !important;
  margin: 0 0.35em 0 0 !important;
  cursor: pointer !important;
  vertical-align: middle !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
/* Latched True: only border/ring → #00FF00 + glow (stays until group changes) */
#controls input[type="checkbox"]:checked,
#controls input[type="radio"]:checked {
  background: var(--ft-disk) !important;
  background-color: var(--ft-disk) !important;
  border-color: var(--ft-ring-active) !important;
  box-shadow: var(--ft-glow) !important;
  accent-color: #00FF00 !important;
}
/* Optional inner concentric tick ring (hollow center) */
#controls input[type="radio"]:checked {
  /* double-ring look: outer glow ring via box-shadow, solid #00FF00 border */
  box-shadow:
    inset 0 0 0 1.5px rgba(15, 23, 42, 0.95),
    var(--ft-glow) !important;
}
/* Radio / toggle chips: single-row body height */
#controls fieldset,
#controls .wrap,
#controls [data-testid="radio-group"],
#controls .form {
  gap: 0.25rem !important;
}
#controls fieldset label,
#controls label:has(input[type="radio"]),
#controls label:has(input[type="checkbox"]),
#controls .wrap button,
#controls button.selected,
#controls [role="radio"] {
  min-height: var(--ft-btn-row-h) !important;
  max-height: var(--ft-btn-row-h) !important;
  height: var(--ft-btn-row-h) !important;
  padding: 0 0.4rem !important;
  margin: 0 !important;
  line-height: 1.15 !important;
  font-size: 0.72rem !important;
  display: inline-flex !important;
  align-items: center !important;
  box-sizing: border-box !important;
}
/* Outer chips always neutral (never full green body) */
#controls .wrap button.selected,
#controls button.selected,
#controls [role="radio"][aria-checked="true"],
#controls label:has(input[type="radio"]:checked),
#controls label:has(input[type="checkbox"]:checked),
#controls fieldset label:has(input:checked),
#controls [data-testid="radio-group"] label:has(input:checked),
#controls .wrap button,
#controls fieldset label {
  background: rgba(15, 23, 42, 0.45) !important;
  background-color: rgba(15, 23, 42, 0.45) !important;
  border-color: rgba(100, 116, 139, 0.4) !important;
  color: #e2e8f0 !important;
  box-shadow: none !important;
}
/* Custom circle span (if Gradio draws one) — ring only when active */
#controls label:has(input:checked) > span:first-child,
#controls [role="radio"][aria-checked="true"]::before {
  width: var(--ft-circle-size) !important;
  height: var(--ft-circle-size) !important;
  border-radius: 50% !important;
  background: var(--ft-disk) !important;
  border: var(--ft-ring-width) solid var(--ft-ring-active) !important;
  box-shadow: var(--ft-glow) !important;
  color: transparent !important;
}
/* Seed status takes remaining column height */
#status-md {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  max-height: none !important;
  overflow: auto !important;
  margin: 0 !important;
  padding: 0.35rem 0.45rem !important;
  color: #cbd5e1 !important;
  font-size: 0.72rem !important;
}
#status-md table {
  width: 100%;
  font-size: 0.7rem;
}
#status-md h1, #status-md h2, #status-md h3 {
  margin: 0.15rem 0 0.35rem 0 !important;
  font-size: 0.85rem !important;
  color: #e2e8f0 !important;
}

/* Images fill viewports without scrollbars */
#workspace .image-container, #workspace img {
  max-height: 100% !important;
  object-fit: contain !important;
}
#viewport-grid .image-container,
#viewport-grid img {
  max-height: 100% !important;
  object-fit: contain !important;
}
.viewport-title {
  color: #64748b !important;
  font-size: 0.7rem !important;
  margin: 0 0 0.15rem 0 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
#hero-caption {
  color: #64748b !important;
  font-size: 0.68rem !important;
  margin: 0.15rem 0 0 0 !important;
}
"""

ABOUT_HTML = f"""
<div class="layer-panel" style="color:#cbd5e1;font-size:0.78rem;line-height:1.45;height:100%;overflow:auto;">
  <b style="color:#e2e8f0;">Photon Seed Asteroids</b> — trajectoid shells + VQC quaternion/OAM
  + live <code>oam_flux</code> Hopf lattice.<br/><br/>
  <b>Shell</b>: 3D shaved sphere from rolling SO(3) + contact trench (Nature-style trajectoids).<br/>
  <b>Nut</b>: hybrid q×scale packing → LG imprint → flywheel flux deposition.<br/>
  <b>Channel</b>: Kolmogorov turbulence scorecard (F, OAM fidelity, Strehl…).<br/>
  <b>Recover</b>: digital / photonic / hybrid with CRC-8.<br/><br/>
  Hero figure: trajectoid bodies + theory/experiment rolling paths
  (Sobolev et al. lineage).<br/><br/>
  <a href="{GITHUB}" style="color:#38bdf8;">GitHub</a> ·
  <a href="{HF_SPACE}" style="color:#38bdf8;">HF Space</a>
</div>
"""


def _empty_imgs():
    b = blank_rgb(280, 360)
    return b, b, b, blank_rgb(400, 220), b, blank_rgb(200, 360), ABOUT_HTML


def _as_on(value) -> bool:
    """Map oval On/Off radios (or legacy bool) → bool latched state."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in ("on", "true", "1", "yes")


def run_ui(
    payload,
    seed,
    turbulence,
    n_steps,
    use_tpt,
    build_3d,
    force_stub,
    slice_frac,
    slice_plane,
    show_slice,
    view,
):
    try:
        out = run_pipeline(
            payload=payload,
            seed=int(seed),
            turbulence=float(turbulence),
            n_steps=int(n_steps),
            use_tpt=_as_on(use_tpt),
            build_3d=_as_on(build_3d),
            force_stub=_as_on(force_stub),
            slice_frac=float(slice_frac),
            slice_plane=str(slice_plane or "z"),
            show_slice=_as_on(show_slice),
        )
        return (
            out["img_shell"],
            out["img_radial"],
            out["img_field"],
            out["img_path"],
            out["img_metrics"],
            out["img_trace"],
            out["status_md"],
            out.get("shell"),
        )
    except Exception as exc:
        logger.exception("pipeline failed")
        err = blank_rgb()
        md = f"### Error\n```\n{exc!r}\n```"
        return err, err, err, err, err, err, md, None


def update_slice_ui(shell, slice_frac, slice_plane, show_slice):
    """Live matrix-slice replot (no full pipeline)."""
    return replot_shell_only(
        shell,
        slice_frac=float(slice_frac) if slice_frac is not None else 0.5,
        slice_plane=str(slice_plane or "z"),
        show_slice=_as_on(show_slice),
    )


def play_scan_ui(shell, slice_plane, n_frames, ping_pong):
    """Synced axial scan: shell + radial + path GIFs at the same pace."""
    n = int(n_frames) if n_frames is not None else 14
    n = max(8, min(24, n))
    g_shell, g_radial, g_path, msg = animate_matrix_scan(
        shell,
        slice_plane=str(slice_plane or "z"),
        n_frames=n,
        # Default off in UI too — ping-pong reads as bounce/glitch
        ping_pong=_as_on(ping_pong),
        duration_ms=90,
    )
    # Gradio Image autoplays GIF paths; fall back to blank if missing
    return (
        g_shell if g_shell else blank_rgb(300, 360),
        g_radial if g_radial else blank_rgb(300, 360),
        g_path if g_path else blank_rgb(400, 220),
        msg,
    )


def build_app() -> gr.Blocks:
    hero = asset_path("trajectoid_paths.png")

    with gr.Blocks(
        title="flux_trajectoid",
        analytics_enabled=False,
    ) as demo:
        # ---- Top navigation ----
        with gr.Row(elem_id="nav-bar"):
            gr.HTML(
                '<span class="nav-title">✦ flux_trajectoid</span>'
            )
            btn_demo = gr.Button(
                "Demo", size="sm", elem_classes=["nav-tab", "nav-active"]
            )
            btn_ref = gr.Button("Reference", size="sm", elem_classes=["nav-tab"])
            btn_about = gr.Button("About", size="sm", elem_classes=["nav-tab"])
            gr.HTML(
                f'<span id="nav-meta">'
                f'<a href="{GITHUB}" style="color:#64748b;text-decoration:none;">GitHub</a>'
                f' · <a href="{HF_SPACE}" style="color:#64748b;text-decoration:none;">HF Space</a>'
                f"</span>"
            )

        view_state = gr.State("demo")
        shell_state = gr.State(None)

        # ---- Workspace (single page, multi-column) ----
        with gr.Row(elem_id="workspace", equal_height=True):
            # Column 1 — CONTROLS | REFERENCES tabs
            with gr.Column(
                scale=2, min_width=200, elem_classes=["layer-inner"], elem_id="controls"
            ):
                with gr.Tabs(elem_id="col1-tabs", selected=0) as col1_tabs:
                    with gr.Tab("CONTROLS", id=0):
                        with gr.Column(elem_id="controls-top"):
                            # Startup: Matrix slice open · others collapsed
                            with gr.Accordion(
                                "Payload & identity",
                                open=False,
                                elem_classes=["layer-fg"],
                            ):
                                payload = gr.Textbox(
                                    value="Hello from the shell",
                                    label="Payload",
                                    lines=1,
                                    max_lines=1,
                                )
                                seed = gr.Number(
                                    value=42, label="Seed", precision=0
                                )
                            with gr.Accordion(
                                "Channel", open=False, elem_classes=["layer-fg"]
                            ):
                                turbulence = gr.Slider(
                                    0.0,
                                    0.8,
                                    value=0.25,
                                    step=0.05,
                                    label="Turbulence",
                                )
                                n_steps = gr.Slider(
                                    4, 32, value=12, step=1, label="Channel steps"
                                )
                            with gr.Accordion(
                                "Shell / lattice",
                                open=False,
                                elem_classes=["layer-fg"],
                            ):
                                use_tpt = gr.Radio(
                                    choices=["On", "Off"],
                                    value="On",
                                    label="TPT closure",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                                build_3d = gr.Radio(
                                    choices=["On", "Off"],
                                    value="On",
                                    label="3D shell",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                                force_stub = gr.Radio(
                                    choices=["On", "Off"],
                                    value="On",
                                    label="Fast stub lattice",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                            with gr.Accordion(
                                "Matrix slice",
                                open=True,
                                elem_classes=["layer-fg"],
                            ):
                                show_slice = gr.Radio(
                                    choices=["On", "Off"],
                                    value="On",
                                    label="Show green slice",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                                slice_plane = gr.Radio(
                                    choices=["x", "y", "z"],
                                    value="z",
                                    label="Slice plane",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                                slice_frac = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=1.0,
                                    step=0.02,
                                    label="Slice position",
                                )
                                scan_frames = gr.Slider(
                                    8,
                                    24,
                                    value=24,
                                    step=1,
                                    label="Scan frames",
                                )
                                scan_ping = gr.Radio(
                                    choices=["On", "Off"],
                                    value="On",
                                    label="Ping-pong scan (can look bouncy)",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                                scan_btn = gr.Button(
                                    "▶ Play matrix scan",
                                    size="sm",
                                    elem_id="scan-btn",
                                )
                            run_btn = gr.Button(
                                "Build · Propagate · Recover",
                                elem_id="run-btn",
                                size="sm",
                            )
                        status = gr.Markdown(
                            "### Seed status\n_Run **Build** to fill this panel._",
                            elem_id="status-md",
                            elem_classes=["layer-fg"],
                        )
                    with gr.Tab("REFERENCES", id=1):
                        with gr.Column(elem_id="panel-refs"):
                            gr.Markdown(
                                '<p class="viewport-title">Trajectoid gallery</p>'
                            )
                            hero_img = gr.Image(
                                value=hero,
                                label=None,
                                show_label=False,
                                height=420,
                                elem_classes=["layer-fg"],
                            )
                            gr.Markdown(
                                '<p id="hero-caption">Trajectoid shapes + theory/experiment paths · used as HF Space visual language</p>',
                            )

            # Columns 2–3: balanced 2×N viewport grid
            #   row heights equal (shared horizontals)
            #   col widths 4 : 3 (different widths OK)
            #   [ shell  | path     ]
            #   [ radial | scorecard]
            #   [ field  | trace    ]
            with gr.Column(
                scale=7,
                min_width=560,
                elem_id="viewport-grid",
            ):
                with gr.Row(
                    equal_height=True,
                    elem_classes=["vp-row"],
                    elem_id="vp-row-top",
                ):
                    with gr.Column(
                        scale=4,
                        min_width=280,
                        elem_classes=["layer-inner", "vp-cell"],
                        elem_id="vp-shell",
                    ):
                        gr.Markdown(
                            '<p class="viewport-title">3D shell · contact path · matrix slice</p>'
                        )
                        img_shell = gr.Image(
                            value=blank_rgb(300, 360),
                            label=None,
                            show_label=False,
                            height=220,
                            elem_classes=["vp-plot"],
                        )
                    with gr.Column(
                        scale=3,
                        min_width=200,
                        elem_classes=["layer-inner", "vp-cell"],
                        elem_id="vp-path",
                    ):
                        gr.Markdown(
                            '<p class="viewport-title">Rolling path (Nature-style)</p>'
                        )
                        img_path = gr.Image(
                            value=blank_rgb(400, 220),
                            label=None,
                            show_label=False,
                            height=220,
                            elem_classes=["vp-plot"],
                        )
                with gr.Row(
                    equal_height=True,
                    elem_classes=["vp-row"],
                    elem_id="vp-row-mid",
                ):
                    with gr.Column(
                        scale=4,
                        min_width=280,
                        elem_classes=["layer-inner", "vp-cell"],
                        elem_id="vp-radial",
                    ):
                        gr.Markdown(
                            '<p class="viewport-title">Radial trench / shave</p>'
                        )
                        img_radial = gr.Image(
                            value=blank_rgb(300, 360),
                            label=None,
                            show_label=False,
                            height=220,
                            elem_classes=["vp-plot"],
                        )
                    with gr.Column(
                        scale=3,
                        min_width=200,
                        elem_classes=["layer-inner", "vp-cell"],
                        elem_id="vp-score",
                    ):
                        gr.Markdown('<p class="viewport-title">Scorecard</p>')
                        img_metrics = gr.Image(
                            value=blank_rgb(260, 360),
                            label=None,
                            show_label=False,
                            height=220,
                            elem_classes=["vp-plot"],
                        )
                with gr.Row(
                    equal_height=True,
                    elem_classes=["vp-row"],
                    elem_id="vp-row-bot",
                ):
                    with gr.Column(
                        scale=4,
                        min_width=280,
                        elem_classes=["layer-inner", "vp-cell"],
                        elem_id="vp-field",
                    ):
                        gr.Markdown(
                            '<p class="viewport-title">Protected OAM field</p>'
                        )
                        img_field = gr.Image(
                            value=blank_rgb(260, 360),
                            label=None,
                            show_label=False,
                            height=220,
                            elem_classes=["vp-plot"],
                        )
                    with gr.Column(
                        scale=3,
                        min_width=200,
                        elem_classes=["layer-inner", "vp-cell"],
                        elem_id="vp-trace",
                    ):
                        gr.Markdown('<p class="viewport-title">Fidelity trace</p>')
                        img_trace = gr.Image(
                            value=blank_rgb(200, 360),
                            label=None,
                            show_label=False,
                            height=220,
                            elem_classes=["vp-plot"],
                        )

        # Hidden reference / about swap targets (still one page — toggle visibility via content)
        ref_panel = gr.Image(
            value=asset_path("shell_construction.png"),
            visible=False,
            label="Shell construction reference",
        )

        outs = [img_shell, img_radial, img_field, img_path, img_metrics, img_trace, status]
        ctrl_inputs = [
            payload,
            seed,
            turbulence,
            n_steps,
            use_tpt,
            build_3d,
            force_stub,
            slice_frac,
            slice_plane,
            show_slice,
            view_state,
        ]

        run_btn.click(
            fn=run_ui,
            inputs=ctrl_inputs,
            outputs=outs + [shell_state],
        )

        # Live matrix slice: replot 3D shell only (no full pipeline)
        slice_inputs = [shell_state, slice_frac, slice_plane, show_slice]
        for ctrl in (slice_frac, slice_plane, show_slice):
            ctrl.change(
                fn=update_slice_ui,
                inputs=slice_inputs,
                outputs=[img_shell],
            )

        # Synced axial scan: shell + radial + path (same timeline)
        scan_btn.click(
            fn=play_scan_ui,
            inputs=[shell_state, slice_plane, scan_frames, scan_ping],
            outputs=[img_shell, img_radial, img_path, status],
        )

        def show_ref(shell_img, radial_img, field_img, path_img, metrics_img, trace_img, st):
            # Open col-1 REFERENCES tab + construction figure
            ref = asset_path("shell_construction.png")
            hero_p = asset_path("trajectoid_paths.png")
            md = (
                "### Reference figures\n"
                "**(gallery)** Trajectoid polyhedra + rolling paths "
                "(theory black / experiment blue).\n\n"
                "**(construction)** Inclined path T, potential surface, shell/core/remnant "
                "with shaved region — the geometric language of Photon Seed Asteroids.\n"
            )
            return (
                shell_img,
                radial_img,
                field_img,
                path_img,
                metrics_img,
                trace_img,
                md,
                gr.update(value=ref or hero_p),
                gr.update(selected=1),  # REFERENCES tab
            )

        def show_about(shell_img, radial_img, field_img, path_img, metrics_img, trace_img, st):
            return (
                shell_img,
                radial_img,
                field_img,
                path_img,
                metrics_img,
                trace_img,
                ABOUT_HTML,
                gr.update(value=asset_path("trajectoid_paths.png")),
                gr.update(selected=0),  # CONTROLS tab
            )

        btn_ref.click(
            fn=show_ref,
            inputs=outs,
            outputs=outs + [hero_img, col1_tabs],
        )
        btn_about.click(
            fn=show_about,
            inputs=outs,
            outputs=outs + [hero_img, col1_tabs],
        )
        btn_demo.click(
            fn=lambda *a: (a[-1], gr.update(selected=0)),
            inputs=outs,
            outputs=[status, col1_tabs],
        )

        # Auto-run once on load for immediate visual
        demo.load(
            fn=run_ui,
            inputs=ctrl_inputs,
            outputs=outs + [shell_state],
        )

    return demo


def _make_theme():
    try:
        return gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ).set(
            body_background_fill="#070b14",
            block_background_fill="rgba(15,23,42,0.3)",
            border_color_primary="rgba(148,163,184,0.2)",
        )
    except Exception:
        return None


# Gradio injects launch(js=...) as <script>textContent</script> — must be an
# IIFE (plain () => {} is never called). Slider fill + main nav active tab.
SLIDER_FILL_JS = """
(() => {
  /* theme_default_tabs: main nav active underline/text */
  function bindNavTabs() {
    const bar = document.querySelector('#nav-bar');
    if (!bar) return;
    bar.querySelectorAll('button').forEach((btn) => {
      if (btn.dataset.ftNavBound === '1') return;
      btn.dataset.ftNavBound = '1';
      btn.addEventListener('click', () => {
        bar.querySelectorAll('button').forEach((b) => {
          b.classList.remove('nav-active', 'primary', 'selected');
        });
        btn.classList.add('nav-active');
      });
    });
  }

  function pctOf(el) {
    const min = parseFloat(el.min || '0');
    const max = parseFloat(el.max || '100');
    const val = parseFloat(el.value);
    const den = max - min;
    if (!isFinite(den) || den === 0) return 0;
    return Math.max(0, Math.min(100, ((val - min) / den) * 100));
  }
  function ensureShell(el) {
    if (el.closest('.ft-slider-shell')) return el.closest('.ft-slider-shell');
    const shell = document.createElement('div');
    shell.className = 'ft-slider-shell';
    const rail = document.createElement('div');
    rail.className = 'ft-slider-rail';
    const fill = document.createElement('div');
    fill.className = 'ft-slider-fill';
    rail.appendChild(fill);
    const parent = el.parentNode;
    if (!parent) return null;
    parent.insertBefore(shell, el);
    shell.appendChild(rail);
    shell.appendChild(el);
    return shell;
  }
  function paint(el) {
    const shell = el.closest('.ft-slider-shell') || ensureShell(el);
    if (!shell) return;
    const fill = shell.querySelector('.ft-slider-fill');
    if (!fill) return;
    // green glowing line from 0 → knob
    fill.style.width = pctOf(el) + '%';
    el.style.setProperty('--slider-color', 'transparent');
    el.style.setProperty('--range_progress', '0%');
  }
  function bindAll() {
    const scope = document.querySelector('#controls') || document;
    scope.querySelectorAll('input[type="range"]').forEach((el) => {
      if (el.dataset.ftSliderBound === '1') {
        paint(el);
        return;
      }
      el.dataset.ftSliderBound = '1';
      ensureShell(el);
      const upd = () => paint(el);
      el.addEventListener('input', upd, { passive: true });
      el.addEventListener('change', upd, { passive: true });
      el.addEventListener('pointermove', () => {
        if (el.matches(':active')) paint(el);
      }, { passive: true });
      const desc = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype, 'value'
      );
      if (desc && desc.set) {
        const orig = desc.set;
        Object.defineProperty(el, 'value', {
          get: desc.get,
          set(v) {
            orig.call(this, v);
            paint(this);
          },
          configurable: true,
        });
      }
      paint(el);
    });
  }
  // Gradio may inject this script before components mount
  const start = () => {
    bindAll();
    bindNavTabs();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
  const obs = new MutationObserver(() => start());
  obs.observe(document.documentElement, { childList: true, subtree: true });
  setInterval(start, 400);
  setTimeout(start, 0);
  setTimeout(start, 250);
  setTimeout(start, 1000);
})();
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    demo = build_app()
    theme = _make_theme()
    # Gradio 4.x: css/theme on Blocks; 5/6: often on launch — try launch first
    queued = demo.queue(default_concurrency_limit=1)
    try:
        queued.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            theme=theme,
            css=CUSTOM_CSS,
            js=SLIDER_FILL_JS,
        )
    except TypeError:
        # Fallback: inject CSS via head if launch rejects css/js
        demo.css = CUSTOM_CSS  # type: ignore[attr-defined]
        try:
            queued.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True,
                js=SLIDER_FILL_JS,
            )
        except TypeError:
            queued.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True,
            )
