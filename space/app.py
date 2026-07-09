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
    replot_scan_suite,
    replot_shell_only,
    run_pipeline,
)

# Keep import alias clear for suite return

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme / CSS — fixed single page, glass layers @ 0.3 opacity
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/*
 * Target host: 1920×1080 (and any full-viewport iframe).
 *
 * Hub page: <iframe class="space-iframe grow"> — Gradio iFrameResizer often
 * pins height to content. We:
 *  1) Size --ft-app-h to the real window (JS) so the iframe can grow
 *  2) Pin #nav-bar, #controls, and each #vp-* with position:fixed pixel boxes
 *     so Gradio flex nesting cannot collapse the 2×2 plots to empty
 *
 * Layout:
 *   [ controls ~300px ] [ shell 2.2fr | path 1fr ]
 *                       [ radial 2.2fr | score 1fr ]
 */
:root {
  --ft-nav-h: 48px;
  --ft-app-h: 100vh;
  --ft-ctrl-w: 300px;
  --ft-gap: 0.4rem;
}
html, body {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: #070b14 !important;
  overflow-x: hidden !important;
  overflow-y: hidden !important;
  /* min-height driven by #ft-spacer / JS — do NOT max-height clip */
  min-height: var(--ft-app-h) !important;
  height: var(--ft-app-h) !important;
}
gradio-app, #root {
  display: block !important;
  width: 100% !important;
  min-height: var(--ft-app-h) !important;
  height: var(--ft-app-h) !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  background: #070b14 !important;
}
.gradio-container,
.gradio-container.fillable,
.gradio-container.app {
  width: 100% !important;
  max-width: 100% !important;
  min-height: var(--ft-app-h) !important;
  height: var(--ft-app-h) !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  background: #070b14 !important;
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif !important;
  position: relative !important;
}
/* In-flow spacer: tells iFrameResizer / content height the real target */
#ft-spacer {
  display: block !important;
  width: 1px !important;
  height: var(--ft-app-h) !important;
  min-height: var(--ft-app-h) !important;
  margin: 0 !important;
  padding: 0 !important;
  pointer-events: none !important;
  opacity: 0 !important;
  overflow: hidden !important;
}
/* Flatten Gradio wrappers so fixed shell can cover the iframe */
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container > .main,
.gradio-container .app,
.gradio-container > .wrap,
.gradio-container > .contain {
  margin: 0 !important;
  padding: 0 !important;
  gap: 0 !important;
  width: 100% !important;
  min-height: 0 !important;
  height: auto !important;
  overflow: visible !important;
  background: transparent !important;
}
footer,
.footer,
#footer,
.svelte-1ipelgc /* gradio footer class variants */ {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
}

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

/* Top navigation — fixed to top of iframe viewport */
#nav-bar {
  display: flex !important;
  align-items: center;
  gap: 0.85rem;
  padding: 0.35rem 0.75rem 0;
  background: rgba(15, 23, 42, 0.98);
  border-bottom: none;
  min-height: var(--ft-nav-h) !important;
  max-height: var(--ft-nav-h) !important;
  height: var(--ft-nav-h) !important;
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  width: 100% !important;
  z-index: 1000 !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
  margin: 0 !important;
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

/* Controls + plots-col pinned; viewports placed as fixed 2×2 by CSS fallback + JS */
#workspace {
  position: relative !important;
  width: 100% !important;
  min-height: calc(100vh - var(--ft-nav-h)) !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: #070b14 !important;
  overflow: visible !important;
}
/* ---- Left controls pane ---- */
#controls {
  position: fixed !important;
  top: calc(var(--ft-nav-h) + var(--ft-gap)) !important;
  left: var(--ft-gap) !important;
  width: var(--ft-ctrl-w) !important;
  max-width: var(--ft-ctrl-w) !important;
  min-width: var(--ft-ctrl-w) !important;
  bottom: var(--ft-gap) !important;
  z-index: 40 !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  display: flex !important;
  flex-direction: column !important;
  box-sizing: border-box !important;
  padding: 0.35rem 0.45rem !important;
  background: rgba(15, 23, 42, 0.96) !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
  border-radius: 10px !important;
  scrollbar-width: thin;
  visibility: visible !important;
  opacity: 1 !important;
}
/*
 * Four plot panels (#img-shell/path/radial/metrics) are position:fixed
 * 2×2 boxes over the plots area. Self-contained HTML (title+img) inside.
 * Escape ALL Gradio flex nesting — that was zeroing paint.
 */
#plots-col,
#plots-placeholder-wrap {
  visibility: hidden !important;
  pointer-events: none !important;
  opacity: 0 !important;
  min-height: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
}
/* Fixed 2×2 — CSS fallback (JS refines with measured nav/ctrl widths) */
#img-shell,
#img-path,
#img-radial,
#img-metrics {
  position: fixed !important;
  z-index: 55 !important;
  display: block !important;
  box-sizing: border-box !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  opacity: 1 !important;
  visibility: visible !important;
  pointer-events: auto !important;
  background: transparent !important;
  border: none !important;
  transform: none !important;
  filter: none !important;
  min-height: 200px !important;
}
#img-shell {
  top: calc(var(--ft-nav-h) + var(--ft-gap)) !important;
  left: calc(var(--ft-ctrl-w) + 2 * var(--ft-gap)) !important;
  width: calc((100vw - var(--ft-ctrl-w) - 4 * var(--ft-gap)) * 0.688) !important;
  height: calc((100vh - var(--ft-nav-h) - 3 * var(--ft-gap)) / 2) !important;
}
#img-path {
  top: calc(var(--ft-nav-h) + var(--ft-gap)) !important;
  left: calc(var(--ft-ctrl-w) + 3 * var(--ft-gap) + (100vw - var(--ft-ctrl-w) - 4 * var(--ft-gap)) * 0.688) !important;
  width: calc((100vw - var(--ft-ctrl-w) - 4 * var(--ft-gap)) * 0.312) !important;
  height: calc((100vh - var(--ft-nav-h) - 3 * var(--ft-gap)) / 2) !important;
}
#img-radial {
  top: calc(var(--ft-nav-h) + 2 * var(--ft-gap) + (100vh - var(--ft-nav-h) - 3 * var(--ft-gap)) / 2) !important;
  left: calc(var(--ft-ctrl-w) + 2 * var(--ft-gap)) !important;
  width: calc((100vw - var(--ft-ctrl-w) - 4 * var(--ft-gap)) * 0.688) !important;
  height: calc((100vh - var(--ft-nav-h) - 3 * var(--ft-gap)) / 2) !important;
}
#img-metrics {
  top: calc(var(--ft-nav-h) + 2 * var(--ft-gap) + (100vh - var(--ft-nav-h) - 3 * var(--ft-gap)) / 2) !important;
  left: calc(var(--ft-ctrl-w) + 3 * var(--ft-gap) + (100vw - var(--ft-ctrl-w) - 4 * var(--ft-gap)) * 0.688) !important;
  width: calc((100vw - var(--ft-ctrl-w) - 4 * var(--ft-gap)) * 0.312) !important;
  height: calc((100vh - var(--ft-nav-h) - 3 * var(--ft-gap)) / 2) !important;
}
/* Gradio HTML chrome inside fixed panel */
#img-shell .html-container,
#img-path .html-container,
#img-radial .html-container,
#img-metrics .html-container,
#img-shell .prose,
#img-path .prose,
#img-radial .prose,
#img-metrics .prose,
#img-shell .ft-vp-html,
#img-path .ft-vp-html,
#img-radial .ft-vp-html,
#img-metrics .ft-vp-html,
#img-shell .ft-vp-panel,
#img-path .ft-vp-panel,
#img-radial .ft-vp-panel,
#img-metrics .ft-vp-panel {
  width: 100% !important;
  height: 100% !important;
  min-height: 100% !important;
  max-height: 100% !important;
  display: block !important;
  opacity: 1 !important;
  visibility: visible !important;
  overflow: hidden !important;
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  box-sizing: border-box !important;
}
#img-shell .pending,
#img-path .pending,
#img-radial .pending,
#img-metrics .pending {
  opacity: 1 !important;
}
#img-shell img,
#img-path img,
#img-radial img,
#img-metrics img,
#img-shell .ft-vp-img,
#img-path .ft-vp-img,
#img-radial .ft-vp-img,
#img-metrics .ft-vp-img {
  position: relative !important;
  z-index: 62 !important;
  display: block !important;
  max-width: 100% !important;
  max-height: 100% !important;
  width: auto !important;
  height: auto !important;
  object-fit: contain !important;
  opacity: 1 !important;
  visibility: visible !important;
}

/* Column 1: compact controls stack
 * IMPORTANT: do NOT force display on .tabitem — Gradio 5 hides inactive
 * panels with inline display:none. Forcing flex + hiding :not(.selected)
 * wiped the entire CONTROLS body (panels never get .selected; only buttons do).
 */
#controls {
  display: flex !important;
  flex-direction: column !important;
  padding: 0.3rem 0.45rem !important;
  gap: 0.2rem !important;
  min-height: 0 !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
}
#controls > .wrap,
#controls > .form,
#col1-tabs,
#controls-top {
  display: flex !important;
  flex-direction: column !important;
  min-width: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
#controls-top {
  flex: 0 0 auto !important;
  gap: 0.18rem !important;
  min-height: 0 !important;
  overflow: visible !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#controls-top > .wrap,
#controls-top > .form,
#controls-top > div,
#controls-top .block,
#controls-top .accordion,
#controls-top details {
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#controls .viewport-title {
  margin: 0 0 0.1rem 0 !important;
  font-size: 0.65rem !important;
  line-height: 1.1 !important;
}
/* Column-1: CONTROLS | REFERENCES — theme_default_tabs
 * Gradio 5/6: .tab-wrapper > .tab-container > button.selected
 */
#col1-tabs {
  --color-accent: var(--ft-tab-active);
  margin: 0 !important;
  padding: 0 !important;
  gap: 0 !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
/* Tab button row only */
#col1-tabs .tab-wrapper,
#controls #col1-tabs .tab-wrapper {
  flex: 0 0 auto !important;
  height: auto !important;
  min-height: 1.5rem !important;
  padding-bottom: 0.15rem !important;
  margin-bottom: 0.35rem !important;
  background: transparent !important;
}
#col1-tabs .tab-container,
#controls #col1-tabs .tab-container {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: flex-end !important;
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
/* Idle tabs */
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
/* active_tab → #00FF00 */
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
#col1-tabs .tab-container button.selected:after,
#col1-tabs button.selected:after,
#controls #col1-tabs button.selected:after {
  background-color: var(--ft-tab-active) !important;
  height: 2px !important;
}
/*
 * Tab panels: leave display to Gradio (inline none/block).
 * Only size/scroll when Gradio leaves the panel visible.
 */
#col1-tabs .tabitem,
#col1-tabs [role="tabpanel"] {
  min-height: 0 !important;
  overflow: auto !important;
  padding: 0 !important;
  border: none !important;
  visibility: visible !important;
}
#col1-tabs .tabitem > .wrap,
#col1-tabs .tabitem > .form,
#col1-tabs .tabitem > div,
#col1-tabs [role="tabpanel"] > .wrap,
#col1-tabs [role="tabpanel"] > div {
  min-width: 0 !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
#panel-refs {
  display: flex !important;
  flex-direction: column !important;
  gap: 0.35rem !important;
  min-height: 0 !important;
  overflow: auto !important;
}
#panel-refs p,
#panel-refs li {
  color: #94a3b8 !important;
  font-size: 0.72rem !important;
  line-height: 1.4 !important;
}
#controls label, #controls span, #controls p {
  color: #94a3b8 !important;
  font-size: 0.68rem !important;
}
/* Text/number only — never paint range/radio (those have their own theme) */
#controls input:not([type="range"]):not([type="radio"]):not([type="checkbox"]),
#controls textarea {
  background: rgba(15, 23, 42, 0.55) !important;
  border-color: rgba(100, 116, 139, 0.4) !important;
  color: #e2e8f0 !important;
  font-size: 0.75rem !important;
  min-height: 1.6rem !important;
  padding: 0.15rem 0.35rem !important;
}
/*
 * theme_default_slider
 *   analog_fill_color     = #00FF00  (0 → knob only)
 *   analog_bar_height     = ~1.5px thin line (was fat full-input paint; −90%)
 *   analog_effect_glowing = True mild (fill + knob)
 *
 * ONLY the .ft-slider-fill layer paints green — never the full input height.
 */
:root {
  --ft-slider-fill: #00FF00;
  --ft-slider-rail: rgba(100, 116, 139, 0.45);
  --ft-slider-h: 1.5px;          /* thin analog line */
  --ft-slider-thumb: 12px;
  --ft-fill-pct: 0%;
  /* very mild glow — not a thick bloom */
  --ft-slider-glow:
    0 0 2px 0.4px rgba(0, 255, 0, 0.7),
    0 0 4px 0.8px rgba(0, 255, 0, 0.35);
}
/* Shell: rail + fill under transparent range */
.ft-slider-shell {
  position: relative !important;
  flex: 1 1 auto !important;
  width: 100% !important;
  min-width: 0 !important;
  height: var(--ft-slider-thumb) !important;
  display: block !important;
  margin: 0.3rem 0 0.5rem 0 !important;
  overflow: visible !important;
  --ft-fill-pct: 0%;
}
/* Dim full-width rail */
.ft-slider-shell > .ft-slider-rail {
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
  overflow: visible !important;
}
/* Green glowing fill: 0 → knob (sibling of rail, not clipped) */
.ft-slider-shell > .ft-slider-fill {
  position: absolute !important;
  left: 0 !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  height: var(--ft-slider-h) !important;
  width: var(--ft-fill-pct, 0%) !important;
  max-width: 100% !important;
  border-radius: 999px !important;
  background: #00FF00 !important;
  background-color: #00FF00 !important;
  box-shadow: var(--ft-slider-glow) !important;
  pointer-events: none !important;
  z-index: 1 !important;
  transition: none !important;
}
/* Range on top — fully transparent track (green line is .ft-slider-fill only) */
.ft-slider-shell > input[type="range"],
#controls .ft-slider-shell input[type="range"],
#controls input[type="range"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  position: relative !important;
  z-index: 3 !important;
  width: 100% !important;
  height: var(--ft-slider-thumb) !important;
  min-height: var(--ft-slider-thumb) !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  outline: none !important;
  cursor: pointer !important;
  /* NEVER paint full-height green on the input — that made a fat bar */
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  accent-color: transparent !important;
  --slider-color: transparent !important;
  --color-accent: transparent !important;
}
.ft-slider-shell > input[type="range"]::-webkit-slider-runnable-track,
#controls input[type="range"]::-webkit-slider-runnable-track {
  height: var(--ft-slider-h) !important;
  border-radius: 999px !important;
  border: none !important;
  background: transparent !important;
  background-image: none !important;
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
  border: 1.5px solid #00FF00 !important;
  box-shadow: var(--ft-slider-glow) !important;
  cursor: pointer !important;
  position: relative !important;
  z-index: 4 !important;
}
.ft-slider-shell > input[type="range"]::-moz-range-track,
#controls input[type="range"]::-moz-range-track {
  height: var(--ft-slider-h) !important;
  border-radius: 999px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* Firefox progress kept thin + mild glow */
.ft-slider-shell > input[type="range"]::-moz-range-progress,
#controls input[type="range"]::-moz-range-progress {
  height: var(--ft-slider-h) !important;
  border-radius: 999px !important;
  background: #00FF00 !important;
  border: none !important;
  box-shadow: var(--ft-slider-glow) !important;
}
.ft-slider-shell > input[type="range"]::-moz-range-thumb,
#controls input[type="range"]::-moz-range-thumb {
  width: var(--ft-slider-thumb) !important;
  height: var(--ft-slider-thumb) !important;
  border-radius: 50% !important;
  background: #f8fafc !important;
  border: 1.5px solid #00FF00 !important;
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
/* Play matrix scan — default idle styling; stays in left column flow */
#scan-btn {
  background: rgba(0, 255, 0, 0.12) !important;
  color: #86efac !important;
  border: 1px solid rgba(0, 255, 0, 0.45) !important;
  font-size: 0.72rem !important;
  min-height: 1.55rem !important;
  max-height: 1.75rem !important;
  margin-top: 0.2rem !important;
  margin-bottom: 0.1rem !important;
  box-shadow: none !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease !important;
}
/* button_state=True while scan runs: same theme_default_button_effect_glowing */
#scan-btn.scan-active,
#scan-btn.play-active,
#controls #scan-btn.scan-active {
  border: 1.5px solid var(--ft-ring-active, #00FF00) !important;
  color: #00FF00 !important;
  background: rgba(0, 255, 0, 0.14) !important;
  box-shadow: var(--ft-glow,
    0 0 4px 1px rgba(0, 255, 0, 0.65),
    0 0 10px 2px rgba(0, 255, 0, 0.4),
    0 0 16px 3px rgba(0, 255, 0, 0.22)
  ) !important;
}

/*
 * theme_default_button — single CIRCLE outline
 *   idle:                    slate circle line
 *   button_state=True:       circle line → #00FF00
 *                            + button_effect_glowing=True
 * Latched until another radio in the group is selected.
 */
:root {
  --ft-circle-size: 1.15em;      /* perfect square box → true circle */
  --ft-ring-width: 2px;          /* stroke width */
  --ft-btn-row-h: 1.45rem;
  --ft-ring-idle: rgba(148, 163, 184, 0.75);
  --ft-ring-active: #00FF00;
  /* button_effect_glowing */
  --ft-glow:
    0 0 4px 1px rgba(0, 255, 0, 0.75),
    0 0 10px 2px rgba(0, 255, 0, 0.45),
    0 0 16px 3px rgba(0, 255, 0, 0.25);
}
#controls input[type="checkbox"],
#controls input[type="radio"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  /* perfect circle geometry — never oval */
  width: var(--ft-circle-size) !important;
  height: var(--ft-circle-size) !important;
  min-width: var(--ft-circle-size) !important;
  min-height: var(--ft-circle-size) !important;
  max-width: var(--ft-circle-size) !important;
  max-height: var(--ft-circle-size) !important;
  aspect-ratio: 1 / 1 !important;
  border-radius: 50% !important;
  /* single circle line only — no fill, no inner ring */
  border: var(--ft-ring-width) solid var(--ft-ring-idle) !important;
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  outline: none !important;
  padding: 0 !important;
  margin: 0 0.35em 0 0 !important;
  flex-shrink: 0 !important;
  cursor: pointer !important;
  vertical-align: middle !important;
  box-sizing: border-box !important;
  display: inline-block !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
/* kill Gradio / browser default checked glyphs (dots, images, inner rings) */
#controls input[type="checkbox"]::before,
#controls input[type="radio"]::before,
#controls input[type="checkbox"]::after,
#controls input[type="radio"]::after {
  content: none !important;
  display: none !important;
  background: none !important;
  border: none !important;
  box-shadow: none !important;
}
/* button_state=True → #00FF00 line + button_effect_glowing */
#controls input[type="checkbox"]:checked,
#controls input[type="radio"]:checked,
#controls input[type="checkbox"]:checked:hover,
#controls input[type="radio"]:checked:hover,
#controls input[type="checkbox"]:checked:focus,
#controls input[type="radio"]:checked:focus {
  border-color: var(--ft-ring-active) !important;
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: var(--ft-glow) !important;
  accent-color: #00FF00 !important;
}
/*
 * Radio blocks (Gradio 5): title above, option chips in a horizontal row.
 * Keep rules light so option wraps never collapse to empty gray bars.
 */
#controls .oval-toggle,
#controls .block.oval-toggle {
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
  gap: 0.15rem !important;
  margin: 0 0 0.25rem 0 !important;
  padding: 0 !important;
  width: 100% !important;
  min-height: auto !important;
  height: auto !important;
  overflow: visible !important;
}
#controls .oval-toggle > span,
#controls .oval-toggle > label:first-child,
#controls .oval-toggle .block-label,
#controls .oval-toggle .block-title,
#controls .oval-toggle [data-testid="block-info"] {
  display: block !important;
  width: 100% !important;
  font-size: 0.65rem !important;
  line-height: 1.25 !important;
  margin: 0 !important;
  padding: 0 !important;
  height: auto !important;
  min-height: 0 !important;
  max-height: none !important;
  white-space: normal !important;
}
/* Option row: visible chips */
#controls .oval-toggle .wrap,
#controls .oval-toggle fieldset,
#controls .oval-toggle [data-testid="radio-group"],
#controls .oval-toggle [role="radiogroup"] {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  align-items: center !important;
  gap: 0.3rem 0.4rem !important;
  width: 100% !important;
  min-height: var(--ft-btn-row-h) !important;
  height: auto !important;
  overflow: visible !important;
  margin: 0 !important;
  padding: 0 !important;
}
#controls .oval-toggle label:has(input[type="radio"]),
#controls .oval-toggle label:has(input[type="checkbox"]),
#controls .oval-toggle [role="radio"] {
  display: inline-flex !important;
  align-items: center !important;
  flex: 0 0 auto !important;
  width: auto !important;
  min-height: var(--ft-btn-row-h) !important;
  height: var(--ft-btn-row-h) !important;
  padding: 0 0.4rem !important;
  margin: 0 !important;
  font-size: 0.7rem !important;
  white-space: nowrap !important;
  visibility: visible !important;
  opacity: 1 !important;
}
/* Action buttons: full-width stack inside left column (never float out) */
#controls #scan-btn,
#controls #run-btn,
#controls button#scan-btn,
#controls button#run-btn {
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  flex: 0 0 auto !important;
  align-self: stretch !important;
  box-sizing: border-box !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
}
/* Outer chips always neutral (never full green body — only the circle glows) */
#controls .oval-toggle label.selected,
#controls [role="radio"][aria-checked="true"],
#controls label:has(input[type="radio"]:checked),
#controls label:has(input[type="checkbox"]:checked),
#controls fieldset label:has(input:checked),
#controls [data-testid="radio-group"] label:has(input:checked),
#controls fieldset label,
#controls .oval-toggle label {
  background: rgba(15, 23, 42, 0.45) !important;
  background-color: rgba(15, 23, 42, 0.45) !important;
  border-color: rgba(100, 116, 139, 0.4) !important;
  color: #e2e8f0 !important;
  box-shadow: none !important;
}
/* Gradio-drawn custom control (span / ::before) — single circle outline */
#controls .oval-toggle label > span:first-child:not(:has(*)),
#controls label:has(input[type="radio"]) > span:first-child,
#controls [role="radio"]::before {
  width: var(--ft-circle-size) !important;
  height: var(--ft-circle-size) !important;
  min-width: var(--ft-circle-size) !important;
  min-height: var(--ft-circle-size) !important;
  max-width: var(--ft-circle-size) !important;
  max-height: var(--ft-circle-size) !important;
  aspect-ratio: 1 / 1 !important;
  border-radius: 50% !important;
  box-sizing: border-box !important;
  padding: 0 !important;
  border: var(--ft-ring-width) solid var(--ft-ring-idle) !important;
  background: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  color: transparent !important;
}
/* button_state=True → #00FF00 line + glow */
#controls label:has(input:checked) > span:first-child,
#controls [role="radio"][aria-checked="true"]::before,
#controls .oval-toggle label.selected > span:first-child {
  border-color: var(--ft-ring-active) !important;
  background: transparent !important;
  background-image: none !important;
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

/* Default image fit; viewport cells override with fill rules above */
#workspace .image-container {
  max-height: 100% !important;
}
.viewport-title {
  color: #64748b !important;
  font-size: 0.7rem !important;
  margin: 0 0 0.15rem 0 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
"""

ABOUT_HTML = f"""
<div class="ft-about" style="color:#cbd5e1;font-size:0.78rem;line-height:1.45;height:100%;overflow:auto;">
  <b style="color:#e2e8f0;">Photon Seed Asteroids</b> — trajectoid shells + VQC quaternion/OAM
  + live <code>oam_flux</code> Hopf lattice.<br/><br/>
  <b>Shell</b>: 3D shaved sphere from rolling SO(3) + contact trench (Nature-style trajectoids).<br/>
  <b>Nut</b>: hybrid q×scale packing → LG imprint → flywheel flux deposition.<br/>
  <b>Channel</b>: Kolmogorov turbulence scorecard (F, OAM fidelity, Strehl…).<br/>
  <b>Recover</b>: digital / photonic / hybrid with CRC-8.<br/><br/>
  Geometry lineage: trajectoid bodies + theory/experiment rolling paths
  (Sobolev et al.).<br/><br/>
  <a href="{GITHUB}" style="color:#38bdf8;">GitHub</a> ·
  <a href="{HF_SPACE}" style="color:#38bdf8;">HF Space</a>
</div>
"""


def _empty_imgs():
    b = blank_rgb(280, 360)
    return b, b, b, blank_rgb(400, 220), b, blank_rgb(200, 360), ABOUT_HTML


def _display_image(value, *, height=360, elem_classes=None, **extra):
    """gr.Image for display-only plots — Gradio 4/5/6 keyword-compatible.

    Prefer a concrete pixel height so Gradio allocates real image space
    (height 100% / flex-only often collapses the img to 0 on HF).
    """
    import inspect

    params = inspect.signature(gr.Image.__init__).parameters
    kwargs: dict = {
        "value": value,
        "label": None,
        "show_label": False,
        "interactive": False,
    }
    if height is not None and "height" in params:
        kwargs["height"] = height
    if elem_classes is not None:
        kwargs["elem_classes"] = elem_classes
    # Prefer Gradio 6 `buttons` / `sources`; fall back to Gradio 4/5 flags
    if "buttons" in params:
        kwargs["buttons"] = []
    else:
        if "show_download_button" in params:
            kwargs["show_download_button"] = False
        if "show_share_button" in params:
            kwargs["show_share_button"] = False
        if "show_fullscreen_button" in params:
            kwargs["show_fullscreen_button"] = False
    if "sources" in params:
        kwargs["sources"] = []
    if "container" in params:
        kwargs["container"] = False
    # type=pil is most reliable for initial paint on Gradio 5 Spaces
    if "type" in params and "type" not in extra:
        kwargs["type"] = "pil"
    kwargs.update(
        {
            k: v
            for k, v in extra.items()
            if k in params or k in ("visible", "elem_id", "elem_classes")
        }
    )
    keep = set(params) | {"elem_classes", "elem_id", "visible"}
    kwargs = {k: v for k, v in kwargs.items() if k in keep}
    return gr.Image(**kwargs)


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
            _to_html(out["img_shell"], "shell"),
            _to_html(out["img_radial"], "radial"),
            _to_pil(out["img_field"]),
            _to_html(out["img_path"], "path"),
            _to_html(out["img_metrics"], "metrics"),
            _to_pil(out["img_trace"]),
            out["status_md"],
            out.get("shell"),
        )
    except Exception as exc:
        logger.exception("pipeline failed")
        err_h = _to_html(blank_rgb(), "err")
        err_i = _to_pil(blank_rgb())
        md = f"### Error\n```\n{exc!r}\n```"
        return err_h, err_h, err_i, err_h, err_h, err_i, md, None


def update_slice_ui(shell, slice_frac, slice_plane, show_slice):
    """
    Live replot of all three scan viewports (shell / radial / path).

    Always returns HTML stills at the same (frac, plane) so a plane change
    cancels any playing GIFs and keeps the suite synchronized.
    """
    pl = _normalize_plane(slice_plane)
    # xyz stills use X as representative (play generates full X→Y→Z GIFs)
    if pl == "xyz":
        pl = "x"
    s, r, p = replot_scan_suite(
        shell,
        slice_frac=float(slice_frac) if slice_frac is not None else 0.5,
        slice_plane=pl,
        show_slice=_as_on(show_slice),
    )
    return _to_html(s, "shell"), _to_html(r, "radial"), _to_html(p, "path")


def _normalize_plane(slice_plane) -> str:
    """Coerce Gradio Radio value → x|y|z|xyz."""
    if isinstance(slice_plane, (list, tuple)):
        slice_plane = slice_plane[0] if slice_plane else "z"
    pl = str(slice_plane or "z").strip().lower()
    if pl in ("x", "y", "z", "xyz"):
        return pl
    return "z"


def _gif_to_html(gif_path, stem: str) -> str:
    """Serve an animated GIF in the fixed HTML panel (never flatten to PNG)."""
    import shutil
    import time

    if not gif_path:
        return _to_html(blank_rgb(300, 360), stem)
    src = Path(gif_path)
    if not src.is_file():
        logger.warning("gif missing for %s: %s", stem, gif_path)
        return _to_html(blank_rgb(300, 360), stem)

    # Unique name each play so the browser restarts the animation (X→Y→Z)
    stamp = int(time.time() * 1000) % 1_000_000_000
    dest = _boot_dir() / f"anim_{stem}_{stamp}.gif"
    try:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        else:
            dest = src
    except Exception:
        logger.exception("gif copy failed %s", src)
        dest = src

    # Verify multi-frame (truncated X-only GIFs were the XYZ symptom)
    n_frames = 1
    try:
        from PIL import Image as PILImage

        with PILImage.open(dest) as im:
            n_frames = int(getattr(im, "n_frames", 1) or 1)
        logger.info("scan gif %s frames=%s size=%s", dest.name, n_frames, dest.stat().st_size)
    except Exception:
        logger.exception("gif inspect failed %s", dest)

    try:
        rel = str(dest.resolve().relative_to(Path(__file__).resolve().parent)).replace(
            "\\", "/"
        )
    except ValueError:
        rel = f"assets/boot/{dest.name}"

    url = _file_url(rel)
    # Title stays generic; per-frame titles live inside the GIF raster
    title = {
        "shell": "3D shell · matrix scan",
        "path": "Rolling path · matrix scan",
        "radial": "Radial trench · matrix scan",
    }.get(stem, stem)
    return (
        f'<div class="ft-vp-panel" data-stem="{stem}" data-frames="{n_frames}" '
        'style="box-sizing:border-box;width:100%;height:100%;min-height:280px;'
        "display:flex;flex-direction:column;padding:8px 10px;"
        "background:rgba(15,23,42,0.98);border:1px solid rgba(148,163,184,0.28);"
        "border-radius:10px;overflow:hidden;opacity:1;visibility:visible;"
        'position:relative;z-index:60;">'
        f'<div class="ft-vp-title" style="flex:0 0 auto;color:#94a3b8;font-size:11px;'
        "font-family:IBM Plex Sans,Segoe UI,system-ui,sans-serif;"
        "text-transform:uppercase;letter-spacing:0.06em;margin:0 0 6px 0;"
        f'line-height:1.2;">{title} · {n_frames}f</div>'
        f'<div class="ft-vp-img-wrap" style="flex:1 1 auto;min-height:220px;'
        "display:flex;align-items:center;justify-content:center;"
        'width:100%;background:#0a0f18;border-radius:6px;overflow:hidden;'
        'position:relative;z-index:61;">'
        f'<img class="ft-vp-img" src="{url}" alt="{stem} scan" draggable="false" '
        'style="position:relative;z-index:62;display:block!important;'
        "max-width:100%;max-height:100%;width:auto;height:auto;"
        "min-width:80px;min-height:80px;object-fit:contain;"
        'opacity:1;visibility:visible;border:0;"/>'
        "</div></div>"
    )


def play_scan_ui(shell, slice_plane, n_frames, ping_pong):
    """Synced axial scan: shell + radial + path GIFs at the same pace."""
    if shell is None:
        empty = _to_html(blank_rgb(300, 360), "shell")
        return (
            empty,
            _to_html(blank_rgb(300, 360), "radial"),
            _to_html(blank_rgb(400, 220), "path"),
            "### Scan\n_No shell yet — click **Build** first, then Play._",
        )
    n = int(n_frames) if n_frames is not None else 14
    n = max(8, min(24, n))
    pl = _normalize_plane(slice_plane)
    g_shell, g_radial, g_path, msg = animate_matrix_scan(
        shell,
        slice_plane=pl,
        n_frames=n,
        # Default off in UI too — ping-pong reads as bounce/glitch
        ping_pong=_as_on(ping_pong),
        duration_ms=90,
        out_dir=_boot_dir(),
    )
    return (
        _gif_to_html(g_shell, "shell"),
        _gif_to_html(g_radial, "radial"),
        _gif_to_html(g_path, "path"),
        msg,
    )


def _boot_dir() -> Path:
    d = Path(__file__).resolve().parent / "assets" / "boot"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _as_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    img = np.asarray(arr)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img[..., :3]


def _to_pil(arr_or_path):
    """Convert ndarray / path / PIL → RGB PIL Image."""
    from PIL import Image as PILImage

    if arr_or_path is None:
        return PILImage.fromarray(blank_rgb(360, 420))
    if isinstance(arr_or_path, PILImage.Image):
        return arr_or_path.convert("RGB")
    if isinstance(arr_or_path, (str, Path)):
        # Already an HTML plot fragment — do not re-open
        s = str(arr_or_path)
        if s.lstrip().startswith("<") or s.startswith("data:image"):
            return PILImage.fromarray(blank_rgb(360, 420))
        p = Path(arr_or_path)
        if not p.is_file():
            p2 = Path(__file__).resolve().parent / str(arr_or_path)
            p = p2 if p2.is_file() else p
        if p.is_file():
            try:
                return PILImage.open(p).convert("RGB")
            except Exception:
                return PILImage.fromarray(blank_rgb(360, 420))
        return PILImage.fromarray(blank_rgb(360, 420))
    return PILImage.fromarray(_as_uint8_rgb(np.asarray(arr_or_path)))


def _file_url(rel_path: str) -> str:
    """Same-origin Gradio file URL (verified working on this Space)."""
    import time

    rel = str(rel_path).replace("\\", "/").lstrip("./")
    return f"/gradio_api/file={rel}?v={int(time.time() * 1000)}"


def _persist_plot(arr_or_path, stem: str) -> str:
    """Write plot bytes under assets/boot/live_{stem}.* and return rel path."""
    import shutil

    from PIL import Image as PILImage

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40] or "plot"
    boot = _boot_dir()

    # Pass-through / resolve existing file
    if isinstance(arr_or_path, (str, Path)):
        s = str(arr_or_path)
        if "ft-vp-img" in s or s.lstrip().startswith("<"):
            return ""  # caller handles HTML passthrough
        p = Path(arr_or_path)
        if not p.is_file():
            p2 = Path(__file__).resolve().parent / s
            p = p2 if p2.is_file() else p
        if p.is_file():
            ext = p.suffix.lower() if p.suffix else ".png"
            if ext not in (".png", ".gif", ".jpg", ".jpeg", ".webp"):
                ext = ".png"
            dest = boot / f"live_{safe}{ext}"
            try:
                if p.resolve() != dest.resolve():
                    shutil.copy2(p, dest)
                return f"assets/boot/{dest.name}"
            except Exception:
                logger.exception("copy plot failed %s → %s", p, dest)

    pil = _to_pil(arr_or_path)
    dest = boot / f"live_{safe}.png"
    try:
        pil.save(dest, format="PNG", optimize=True)
    except Exception:
        logger.exception("save plot failed %s", dest)
        # fall back to shipped boot if present
        shipped = boot / f"img_{safe}.png"
        if not shipped.is_file():
            # map stems shell/path/radial/metrics → boot names
            alt = {
                "shell": "img_shell.png",
                "path": "img_path.png",
                "radial": "img_radial.png",
                "metrics": "img_metrics.png",
                "score": "img_metrics.png",
            }.get(safe)
            if alt:
                shipped = boot / alt
        if shipped.is_file():
            return f"assets/boot/{shipped.name}"
        PILImage.fromarray(blank_rgb(360, 420)).save(dest, format="PNG")
    return f"assets/boot/{dest.name}"


def _to_html(arr_or_path, stem: str = "plot") -> str:
    """Viewport plot as HTML <img> using /gradio_api/file=… (not data: URIs).

    data:image URIs often get stripped by Gradio's HTML sanitizer → empty cells.
    File URLs under assets/boot/ are served (200 OK on this Space) and paint.
    Inline styles force size/z-index so outer CSS cannot collapse the layer.
    """
    if isinstance(arr_or_path, str) and (
        "ft-vp-img" in arr_or_path or arr_or_path.lstrip().startswith("<div")
    ):
        return arr_or_path

    # Prefer known shipped boot files when value is already that path
    rel = ""
    if isinstance(arr_or_path, (str, Path)):
        s = str(arr_or_path).replace("\\", "/")
        if s.startswith("assets/boot/") and (Path(__file__).resolve().parent / s).is_file():
            rel = s
    if not rel:
        rel = _persist_plot(arr_or_path, stem)
    if not rel:
        # last resort: shipped name
        guess = {
            "shell": "assets/boot/img_shell.png",
            "path": "assets/boot/img_path.png",
            "radial": "assets/boot/img_radial.png",
            "metrics": "assets/boot/img_metrics.png",
            "score": "assets/boot/img_metrics.png",
        }.get(stem, "assets/boot/img_shell.png")
        rel = guess

    url = _file_url(rel)
    title = {
        "shell": "3D shell · contact path · matrix slice",
        "path": "Rolling path (Nature-style)",
        "radial": "Radial trench / shave",
        "metrics": "Scorecard",
        "score": "Scorecard",
    }.get(stem, stem)
    # Fully self-contained panel: title + image. All sizing/z-index INLINE so
    # Gradio flex / display:contents / empty overlays cannot zero the paint.
    return (
        f'<div class="ft-vp-panel" data-stem="{stem}" '
        'style="box-sizing:border-box;width:100%;height:100%;min-height:280px;'
        "display:flex;flex-direction:column;padding:8px 10px;"
        "background:rgba(15,23,42,0.98);border:1px solid rgba(148,163,184,0.28);"
        'border-radius:10px;overflow:hidden;opacity:1;visibility:visible;'
        'position:relative;z-index:60;">'
        f'<div class="ft-vp-title" style="flex:0 0 auto;color:#94a3b8;font-size:11px;'
        "font-family:IBM Plex Sans,Segoe UI,system-ui,sans-serif;"
        "text-transform:uppercase;letter-spacing:0.06em;margin:0 0 6px 0;"
        f'line-height:1.2;">{title}</div>'
        f'<div class="ft-vp-img-wrap" style="flex:1 1 auto;min-height:220px;'
        "display:flex;align-items:center;justify-content:center;"
        'width:100%;background:#0a0f18;border-radius:6px;overflow:hidden;'
        'position:relative;z-index:61;">'
        f'<img class="ft-vp-img" src="{url}" alt="{stem}" draggable="false" '
        'style="position:relative;z-index:62;display:block!important;'
        "max-width:100%;max-height:100%;width:auto;height:auto;"
        "min-width:80px;min-height:80px;object-fit:contain;"
        'opacity:1;visibility:visible;border:0;"/>'
        "</div></div>"
    )


def _display_html(value, *, stem: str = "plot", elem_id=None, elem_classes=None):
    """gr.HTML plot viewport — file-URL <img>, always frontmost.

    Gradio 5 HTML collapses to 0px inside flex columns unless min_height is set
    (see html/Index.svelte style:min-height). padding=False avoids chrome.
    """
    import inspect

    classes = list(elem_classes or [])
    if "vp-plot" not in classes:
        classes.append("vp-plot")
    if "ft-vp-html" not in classes:
        classes.append("ft-vp-html")
    kwargs = {
        "value": _to_html(value, stem=stem),
        "elem_classes": classes,
        "show_label": False,
    }
    if elem_id is not None:
        kwargs["elem_id"] = elem_id
    params = inspect.signature(gr.HTML.__init__).parameters
    if "min_height" in params:
        kwargs["min_height"] = 320
    if "padding" in params:
        kwargs["padding"] = False
    if "container" in params:
        kwargs["container"] = False
    kwargs = {k: v for k, v in kwargs.items() if k in params or k in ("elem_id", "elem_classes")}
    return gr.HTML(**kwargs)


def _save_rgb_png(arr: np.ndarray, name: str):
    """Write RGB ndarray to assets/boot/<name>.png; return PIL for gr.Image."""
    from PIL import Image as PILImage

    img = _as_uint8_rgb(arr)
    path = _boot_dir() / f"{name}.png"
    pil = PILImage.fromarray(img)
    try:
        pil.save(path, format="PNG")
    except Exception:
        logger.exception("failed to persist boot png %s", name)
    return pil


def _load_boot_value(key: str):
    """PIL Image from shipped assets/boot/<key>.png (or None)."""
    p = _boot_dir() / f"{key}.png"
    if not (p.is_file() and p.stat().st_size > 100):
        return None
    try:
        return _to_pil(p)
    except Exception:
        logger.exception("failed to load boot png %s", key)
        return None


def _startup_plots() -> dict:
    """Seed four viewports on first paint with numpy RGB (never empty frames)."""
    keys = (
        "img_shell",
        "img_radial",
        "img_field",
        "img_path",
        "img_metrics",
        "img_trace",
    )
    # Instant seed from shipped PNGs so viewports never paint empty
    shipped: dict = {}
    for key in keys:
        val = _load_boot_value(key)
        if val is not None:
            shipped[key] = val

    def _with_display_values(out: dict) -> dict:
        for key in keys:
            if key in out and isinstance(out[key], np.ndarray):
                out[key] = _save_rgb_png(out[key], key)
            elif key in out and out[key] is not None:
                out[key] = _to_pil(out[key])
            elif key in shipped:
                out[key] = shipped[key]
        return out

    if len(shipped) == len(keys):
        try:
            out = run_pipeline(
                payload="Hello from the shell",
                seed=42,
                turbulence=0.25,
                n_steps=12,
                use_tpt=True,
                build_3d=True,
                force_stub=True,
                slice_frac=1.0,
                slice_plane="z",
                show_slice=True,
            )
            out = _with_display_values(out)
            logger.info(
                "startup plots ready values=%s",
                {k: (getattr(out.get(k), "shape", None) or out.get(k)) for k in keys},
            )
            return out
        except Exception:
            logger.exception("startup pipeline failed — using shipped boot PNGs")
            shipped["shell"] = None
            shipped["status_md"] = (
                "### Seed status\n_Stub plots loaded — click **Build** to recompute._"
            )
            return shipped

    try:
        out = run_pipeline(
            payload="Hello from the shell",
            seed=42,
            turbulence=0.25,
            n_steps=12,
            use_tpt=True,
            build_3d=True,
            force_stub=True,
            slice_frac=1.0,
            slice_plane="z",
            show_slice=True,
        )
        out = _with_display_values(out)
        logger.info(
            "startup plots ready values=%s",
            {k: (getattr(out.get(k), "shape", None) or out.get(k)) for k in keys},
        )
        return out
    except Exception:
        logger.exception("startup plots failed")
        b = blank_rgb(420, 360)
        fallback: dict = {}
        for key in keys:
            val = _load_boot_value(key)
            if val is not None:
                fallback[key] = val
            else:
                fallback[key] = _save_rgb_png(
                    b if key != "img_path" else blank_rgb(320, 400),
                    key,
                )
        fallback["shell"] = None
        fallback["status_md"] = (
            "### Seed status\n_Startup failed — click **Build**._"
        )
        return fallback


def build_app() -> gr.Blocks:
    # Gradio 5+: css/js/head are set on launch (and on Blocks when supported).
    # Slider fill JS MUST run on page load — head <script> is most reliable on HF.
    blocks_kwargs = dict(
        title="flux_trajectoid",
        analytics_enabled=False,
    )
    # Inject JS only once via head (js + head both running = double observers / flicker)
    optional_block = {
        "css": CUSTOM_CSS,
        "head": f"<script>\n{SLIDER_FILL_JS}\n</script>",
        "fill_height": True,
        "fill_width": True,
    }
    try:
        import inspect as _inspect

        _bp = _inspect.signature(gr.Blocks.__init__).parameters
        for k, v in optional_block.items():
            if k in _bp:
                blocks_kwargs[k] = v
    except Exception:
        pass

    boot = _startup_plots()

    with gr.Blocks(**blocks_kwargs) as demo:
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
        shell_state = gr.State(boot.get("shell"))

        # ---- Workspace: controls | (2×2 plots) ----
        with gr.Row(elem_id="workspace", equal_height=True):
            # Left — CONTROLS | REFERENCES
            with gr.Column(
                scale=3,
                min_width=280,
                                elem_id="controls",
            ):
                with gr.Tabs(elem_id="col1-tabs", selected="controls") as col1_tabs:
                    with gr.Tab("CONTROLS", id="controls"):
                        with gr.Column(elem_id="controls-top"):
                            # Startup: Matrix slice open · others collapsed
                            with gr.Accordion(
                                "Payload & identity",
                                open=False,
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
                                "Channel", open=False
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
                                                            ):
                                show_slice = gr.Radio(
                                    choices=["On", "Off"],
                                    value="On",
                                    label="Green slice",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                                slice_plane = gr.Radio(
                                    choices=["x", "y", "z", "xyz"],
                                    value="z",
                                    label="Plane",
                                    type="value",
                                    elem_classes=["oval-toggle"],
                                )
                                slice_frac = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=1.0,
                                    step=0.02,
                                    label="Position",
                                )
                                scan_frames = gr.Slider(
                                    8,
                                    24,
                                    value=24,
                                    step=1,
                                    label="Frames",
                                )
                                scan_ping = gr.Radio(
                                    choices=["On", "Off"],
                                    value="On",
                                    label="Ping-pong",
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
                            boot.get(
                                "status_md",
                                "### Seed status\n_Run **Build** to fill this panel._",
                            ),
                            elem_id="status-md",
                                                    )
                    with gr.Tab("REFERENCES", id="references"):
                        with gr.Column(elem_id="panel-refs"):
                            gr.Markdown(
                                """
<p class="viewport-title">References</p>

**Trajectoid language** — polyhedra + rolling paths (theory black / experiment blue),
used as the geometric visual language of Photon Seed Asteroids.

**Construction** — inclined path *T*, potential surface, shell / core / remnant
with shaved region (Nature-style SO(3) rolling).

Links: [GitHub](https://github.com/kinaar8340/flux_trajectoid) ·
[HF Space](https://huggingface.co/spaces/kinaar111/flux_trajectoid)
""",
                                                            )

            # Right placeholder (layout occupied by fixed #img-* panels via CSS/JS)
            with gr.Column(
                scale=7,
                min_width=0,
                elem_id="plots-col",
            ):
                gr.HTML(
                    '<div id="plots-placeholder" style="min-height:4px;opacity:0;">'
                    "plots</div>",
                    elem_id="plots-placeholder-wrap",
                )

        # Four plot panels — NOT nested in Gradio flex rows (that collapsed paint).
        # CSS/JS places them as position:fixed 2×2 over the plots area.
        img_shell = _display_html(
            boot["img_shell"],
            stem="shell",
            elem_classes=["vp-plot", "ft-panel"],
            elem_id="img-shell",
        )
        img_path = _display_html(
            boot["img_path"],
            stem="path",
            elem_classes=["vp-plot", "ft-panel"],
            elem_id="img-path",
        )
        img_radial = _display_html(
            boot["img_radial"],
            stem="radial",
            elem_classes=["vp-plot", "ft-panel"],
            elem_id="img-radial",
        )
        img_metrics = _display_html(
            boot["img_metrics"],
            stem="metrics",
            elem_classes=["vp-plot", "ft-panel"],
            elem_id="img-metrics",
        )

        # Outside #workspace grid so they never steal a column track
        img_field = gr.Image(
            value=boot.get("img_field", blank_rgb(260, 360)),
            visible=False,
            label="Protected OAM field",
        )
        img_trace = gr.Image(
            value=boot.get("img_trace", blank_rgb(200, 360)),
            visible=False,
            label="Fidelity trace",
        )
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

        # Live matrix slice: replot shell + radial + path stills (synced)
        # Frac / show_slice: always stills. Plane XYZ: full X→Y→Z sequence.
        slice_inputs = [shell_state, slice_frac, slice_plane, show_slice]
        for ctrl in (slice_frac, show_slice):
            ctrl.change(
                fn=update_slice_ui,
                inputs=slice_inputs,
                outputs=[img_shell, img_radial, img_path],
            )

        # Synced axial scan: shell + radial + path (same timeline)
        # button_state True → green glowing border; False when finished
        def _scan_btn_on():
            return gr.update(
                value="▶ Playing…",
                interactive=False,
                elem_classes=["scan-active"],
            )

        def _scan_btn_off():
            return gr.update(
                value="▶ Play matrix scan",
                interactive=True,
                elem_classes=[],
            )

        def on_slice_plane_change(
            shell, slice_frac, plane, show_slice, n_frames, ping_pong
        ):
            """X/Y/Z → synced stills; XYZ → play full X→Y→Z GIF sequence."""
            pl = _normalize_plane(plane)
            if pl == "xyz":
                return play_scan_ui(shell, "xyz", n_frames, ping_pong)
            s, r, p = update_slice_ui(shell, slice_frac, pl, show_slice)
            return (
                s,
                r,
                p,
                f"### Plane **{pl.upper()}**\n"
                f"_Synced stills — click **Play** to animate this axis._",
            )

        def _plane_btn_prelude(plane):
            # Glow Play only while XYZ sequence is generating
            if _normalize_plane(plane) == "xyz":
                return _scan_btn_on()
            return gr.update()

        def _plane_btn_epilogue(plane):
            if _normalize_plane(plane) == "xyz":
                return _scan_btn_off()
            return gr.update()

        slice_plane.change(
            fn=_plane_btn_prelude,
            inputs=[slice_plane],
            outputs=[scan_btn],
        ).then(
            fn=on_slice_plane_change,
            inputs=[
                shell_state,
                slice_frac,
                slice_plane,
                show_slice,
                scan_frames,
                scan_ping,
            ],
            outputs=[img_shell, img_radial, img_path, status],
        ).then(
            fn=_plane_btn_epilogue,
            inputs=[slice_plane],
            outputs=[scan_btn],
        )

        scan_btn.click(
            fn=_scan_btn_on,
            inputs=None,
            outputs=[scan_btn],
        ).then(
            fn=play_scan_ui,
            inputs=[shell_state, slice_plane, scan_frames, scan_ping],
            outputs=[img_shell, img_radial, img_path, status],
        ).then(
            fn=_scan_btn_off,
            inputs=None,
            outputs=[scan_btn],
        )

        def show_ref(shell_img, radial_img, field_img, path_img, metrics_img, trace_img, st):
            # Open col-1 REFERENCES tab (text only — gallery image removed)
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
                gr.update(selected="references"),
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
                gr.update(selected="controls"),
            )

        btn_ref.click(
            fn=show_ref,
            inputs=outs,
            outputs=outs + [col1_tabs],
        )
        btn_about.click(
            fn=show_about,
            inputs=outs,
            outputs=outs + [col1_tabs],
        )
        btn_demo.click(
            fn=lambda *a: (a[-1], gr.update(selected="controls")),
            inputs=outs,
            outputs=[status, col1_tabs],
        )

        # Re-push HTML plots after hydrate so front layer is never empty.
        _seed = (
            boot.get("img_shell"),
            boot.get("img_path"),
            boot.get("img_radial"),
            boot.get("img_metrics"),
        )

        def _seed_plots():
            return (
                _to_html(_seed[0], "shell"),
                _to_html(_seed[1], "path"),
                _to_html(_seed[2], "radial"),
                _to_html(_seed[3], "metrics"),
            )

        try:
            demo.load(
                fn=_seed_plots,
                inputs=None,
                outputs=[img_shell, img_path, img_radial, img_metrics],
            )
        except Exception:
            logger.exception("demo.load seed_plots failed")

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


# Gradio injects js as script text — IIFE required. Keep this QUIET:
# no MutationObserver on attributes, no setInterval fitScreen (those
# cycled styles and made the startup page flicker through "layers").
SLIDER_FILL_JS = """
(() => {
  if (window.__ftUiInit) return; // single init even if injected twice
  window.__ftUiInit = true;

  function pctOf(el) {
    const min = parseFloat(el.min || '0');
    const max = parseFloat(el.max || '100');
    const val = parseFloat(el.value);
    const den = max - min;
    if (!isFinite(den) || den === 0) return 0;
    return Math.max(0, Math.min(100, ((val - min) / den) * 100));
  }

  const FILL_STYLE = [
    'position:absolute','left:0','top:50%','transform:translateY(-50%)',
    'height:1.5px','min-height:1.5px','max-height:1.5px','border-radius:999px',
    'background:#00FF00','background-color:#00FF00',
    'box-shadow:0 0 2px 0.4px rgba(0,255,0,0.7),0 0 4px 0.8px rgba(0,255,0,0.35)',
    'pointer-events:none','z-index:2','display:block','margin:0','padding:0','border:none',
  ].join(';');
  const RAIL_STYLE = [
    'position:absolute','left:0','right:0','top:50%','transform:translateY(-50%)',
    'height:1.5px','border-radius:999px','background:rgba(100,116,139,0.45)',
    'pointer-events:none','z-index:1',
  ].join(';');

  function ensureShell(el) {
    let shell = el.closest('.ft-slider-shell');
    if (!shell) {
      shell = document.createElement('div');
      shell.className = 'ft-slider-shell';
      shell.style.cssText = 'position:relative;display:block;width:100%;height:14px;margin:0.3rem 0 0.5rem 0;overflow:visible;box-sizing:border-box;';
      const parent = el.parentNode;
      if (!parent) return null;
      parent.insertBefore(shell, el);
      shell.appendChild(el);
    }
    let rail = shell.querySelector(':scope > .ft-slider-rail');
    if (!rail) {
      rail = document.createElement('div');
      rail.className = 'ft-slider-rail';
      shell.insertBefore(rail, shell.firstChild);
    }
    rail.style.cssText = RAIL_STYLE;
    let fill = shell.querySelector(':scope > .ft-slider-fill');
    if (!fill) {
      fill = document.createElement('div');
      fill.className = 'ft-slider-fill';
      shell.insertBefore(fill, el);
    }
    if (fill.nextSibling !== el) shell.insertBefore(fill, el);
    return shell;
  }

  function paint(el) {
    if (!el || el.type !== 'range') return;
    const shell = ensureShell(el);
    if (!shell) return;
    const fill = shell.querySelector(':scope > .ft-slider-fill');
    if (!fill) return;
    const pct = pctOf(el);
    const shellW = shell.clientWidth || 0;
    const thumb = 12;
    const wPx = shellW > 0
      ? Math.max(0, (thumb / 2) + ((shellW - thumb) * pct) / 100)
      : null;
    fill.style.cssText = FILL_STYLE + ';width:' + (
      wPx != null ? wPx.toFixed(2) + 'px' : pct.toFixed(3) + '%'
    );
    el.style.background = 'transparent';
  }

  function bindSliders() {
    document.querySelectorAll('input[type="range"]').forEach((el) => {
      if (!el.closest('.ft-slider-shell')) el.dataset.ftSliderBound = '0';
      if (el.dataset.ftSliderBound === '1') {
        paint(el);
        return;
      }
      el.dataset.ftSliderBound = '1';
      ensureShell(el);
      const upd = () => paint(el);
      el.addEventListener('input', upd, { passive: true });
      el.addEventListener('change', upd, { passive: true });
      paint(el);
    });
  }

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

  function setImp(el, props) {
    if (!el) return;
    Object.keys(props).forEach((k) => {
      const cssKey = k.replace(/[A-Z]/g, (m) => '-' + m.toLowerCase());
      try {
        el.style.setProperty(cssKey, String(props[k]), 'important');
      } catch (_) {
        el.style[k] = props[k];
      }
    });
  }

  function layoutOnce() {
    // Keep Gradio Image nodes in-tree (reparenting wiped image values).
    // Pin #controls + #plots-col; CSS grid places the 2×2 cells.
    const nav = document.querySelector('#nav-bar');
    const navH = (nav && nav.offsetHeight) || 48;
    document.documentElement.style.setProperty('--ft-nav-h', navH + 'px');

    const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
    const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
    const appH = Math.max(vh, 600);
    document.documentElement.style.setProperty('--ft-app-h', appH + 'px');

    const gap = 8;
    const ctrlW = Math.min(300, Math.max(240, Math.floor(vw * 0.16)));
    document.documentElement.style.setProperty('--ft-ctrl-w', ctrlW + 'px');
    const top = navH + gap;
    const paneH = Math.max(200, appH - top - gap);
    const plotsLeft = ctrlW + 2 * gap;
    const plotsW = Math.max(320, vw - plotsLeft - gap);

    // Kill transform traps on ancestors of plot cells (fixed/grid safe)
    document.querySelectorAll(
      '#workspace, #plots-col, #plots-top, #plots-bot, #controls, #vp-shell, #vp-path, #vp-radial, #vp-score'
    ).forEach((el) => {
      setImp(el, {
        transform: 'none',
        filter: 'none',
        perspective: 'none',
        contain: 'none',
      });
      // Walk a few parents
      let p = el.parentElement;
      for (let i = 0; i < 6 && p && p !== document.body; i++) {
        setImp(p, { transform: 'none', filter: 'none', perspective: 'none' });
        p = p.parentElement;
      }
    });

    if (nav) {
      setImp(nav, {
        position: 'fixed', top: '0px', left: '0px', right: '0px',
        width: '100%', height: navH + 'px', zIndex: '1000',
      });
    }

    const ctrl = document.querySelector('#controls');
    if (ctrl) {
      setImp(ctrl, {
        position: 'fixed',
        top: top + 'px',
        left: gap + 'px',
        width: ctrlW + 'px',
        height: paneH + 'px',
        zIndex: '50',
        display: 'flex',
        flexDirection: 'column',
        overflowX: 'hidden',
        overflowY: 'auto',
        visibility: 'visible',
        opacity: '1',
        boxSizing: 'border-box',
        pointerEvents: 'auto',
      });
    }

    // Hide leftover plots-col stub
    const pc = document.querySelector('#plots-col');
    if (pc) {
      setImp(pc, {
        position: 'fixed', top: '0px', left: '0px',
        width: '0px', height: '0px', opacity: '0',
        pointerEvents: 'none', overflow: 'hidden',
      });
    }

    // 2×2 pixel boxes for the four HTML panels (title+img self-contained)
    const rowGap = gap;
    const colGap = gap;
    const rowH = Math.max(160, Math.floor((paneH - rowGap) / 2));
    const leftW = Math.max(180, Math.floor((plotsW - colGap) * 0.688));
    const rightW = Math.max(140, plotsW - leftW - colGap);
    const y0 = top;
    const y1 = top + rowH + rowGap;
    const x0 = plotsLeft;
    const x1 = plotsLeft + leftW + colGap;

    const placePanel = (sel, x, y, w, h) => {
      const el = document.querySelector(sel);
      if (!el) return false;
      // Reparent HTML panels to body → true viewport-fixed (safe for HTML)
      if (el.parentElement !== document.body) {
        document.body.appendChild(el);
      }
      setImp(el, {
        position: 'fixed',
        top: y + 'px',
        left: x + 'px',
        width: w + 'px',
        height: h + 'px',
        right: 'auto',
        bottom: 'auto',
        zIndex: '55',
        display: 'block',
        overflow: 'hidden',
        visibility: 'visible',
        opacity: '1',
        margin: '0px',
        padding: '0px',
        boxSizing: 'border-box',
        pointerEvents: 'auto',
        background: 'transparent',
        border: 'none',
        transform: 'none',
        minHeight: '160px',
      });
      el.querySelectorAll('.html-container, .prose, .ft-vp-panel, .ft-vp-html').forEach((node) => {
        setImp(node, {
          width: '100%', height: '100%', minHeight: '100%',
          display: 'block', opacity: '1', visibility: 'visible',
          overflow: 'hidden', padding: '0px', margin: '0px',
        });
      });
      el.querySelectorAll('img').forEach((img) => {
        setImp(img, {
          position: 'relative', zIndex: '62',
          maxWidth: '100%', maxHeight: '100%',
          width: 'auto', height: 'auto',
          objectFit: 'contain', display: 'block',
          opacity: '1', visibility: 'visible',
        });
      });
      return true;
    };

    const ok = [
      placePanel('#img-shell', x0, y0, leftW, rowH),
      placePanel('#img-path', x1, y0, rightW, rowH),
      placePanel('#img-radial', x0, y1, leftW, rowH),
      placePanel('#img-metrics', x1, y1, rightW, rowH),
    ];
    if (ok.some((v) => !v)) {
      setTimeout(layoutOnce, 250);
    }
  }

  function start() {
    layoutOnce();
    bindNavTabs();
    bindSliders();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
  // Gradio mounts late — a few one-shot retries (no intervals / no flicker loops)
  setTimeout(start, 200);
  setTimeout(layoutOnce, 500);
  setTimeout(layoutOnce, 1200);
  setTimeout(bindSliders, 1000);
  window.addEventListener('resize', () => {
    layoutOnce();
    bindSliders();
  }, { passive: true });
  window.addEventListener('load', layoutOnce, { passive: true });
  // Only watch for new range inputs (childList), never attributes (avoids loops)
  try {
    const mo = new MutationObserver((muts) => {
      for (const m of muts) {
        if (m.addedNodes && m.addedNodes.length) {
          bindSliders();
          // Re-layout once if plot cells just appeared
          for (const n of m.addedNodes) {
            if (n.nodeType === 1 && (
              (n.id && (n.id === 'plots-col' || n.id.startsWith('vp-'))) ||
              (n.querySelector && n.querySelector('#plots-col, #vp-shell'))
            )) {
              layoutOnce();
              break;
            }
          }
          break;
        }
      }
    });
    mo.observe(document.body || document.documentElement, {
      childList: true,
      subtree: true,
    });
  } catch (_) {}
})();
"""


def _launch(demo, *, port: int, theme, css: str, js: str) -> None:
    """Launch with kwargs filtered to the installed Gradio version."""
    import inspect

    # Single head script only — do not also pass js= (double-init flicker)
    try:
        demo.css = css
    except Exception:
        pass
    try:
        demo.head = f"<script>\n{js}\n</script>"
        demo.js = None
    except Exception:
        pass

    queued = demo.queue(default_concurrency_limit=1)
    base = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "share": False,
        "show_error": True,
    }
    space_root = str(Path(__file__).resolve().parent)
    optional = {
        "theme": theme,
        "css": css,
        "head": f"<script>\n{js}\n</script>",
        "ssr": False,
        "ssr_mode": False,
        "show_api": False,
        "allowed_paths": [space_root, str(Path(space_root) / "assets")],
    }
    try:
        params = inspect.signature(queued.launch).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = dict(base)
    for k, v in optional.items():
        if k in params:
            kwargs[k] = v
    queued.launch(**kwargs)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    demo = build_app()
    theme = _make_theme()
    _launch(demo, port=port, theme=theme, css=CUSTOM_CSS, js=SLIDER_FILL_JS)
