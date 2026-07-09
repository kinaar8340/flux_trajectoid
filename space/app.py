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
 * Full-screen layout for HF Spaces hub iframe.
 *
 * Hub page: <iframe class="space-iframe grow"> — CSS grow wants full
 * remaining viewport, but Gradio iFrameResizer often pins height to
 * content. We:
 *  1) Keep an in-flow #ft-spacer at target host height so the iframe grows
 *  2) Pin #nav-bar + #workspace with position:fixed to fill the iframe
 *     viewport (no mid-panel clip, no empty black half)
 */
:root {
  --ft-nav-h: 48px;
  --ft-app-h: 100vh;
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

/*
 * Workspace: fixed under nav — always fills the iframe viewport.
 * Independent of document flow / iFrameResizer content height.
 */
#workspace {
  position: fixed !important;
  top: var(--ft-nav-h) !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  height: auto !important;
  min-height: 0 !important;
  max-height: none !important;
  width: 100% !important;
  display: grid !important;
  grid-template-columns:
    minmax(260px, 1.05fr)
    minmax(0, 2.2fr)
    minmax(0, 1fr) !important;
  grid-template-rows: minmax(0, 1fr) !important;
  align-items: stretch !important;
  gap: 0.35rem !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
  padding: 0.3rem 0.4rem 0.35rem 0.4rem !important;
  margin: 0 !important;
  z-index: 900 !important;
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(59, 130, 246, 0.12), transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 100%, rgba(167, 139, 250, 0.10), transparent 50%),
    #070b14 !important;
}
/* Kill Gradio default column min-widths; fill grid cell */
#workspace .column,
#workspace [class*="column"],
#workspace > div,
#workspace > * {
  min-width: 0 !important;
  min-height: 0 !important;
  max-width: none !important;
  height: 100% !important;
  max-height: 100% !important;
  box-sizing: border-box !important;
}
#controls {
  grid-column: 1 !important;
  min-width: 0 !important;
  height: 100% !important;
  max-height: 100% !important;
  overflow-x: hidden !important;
}
#col-center {
  grid-column: 2 !important;
  min-width: 0 !important;
  height: 100% !important;
  max-height: 100% !important;
}
#col-right {
  grid-column: 3 !important;
  min-width: 0 !important;
  height: 100% !important;
  max-height: 100% !important;
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
}

/* Col-1 scrolls; plot cols do not (fixed half-panels) */
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

/* Column 2 & 3: two equal-height plot cells (shell/radial · path/scorecard) */
#col-center,
#col-right {
  height: 100% !important;
  max-height: 100% !important;
  min-height: 0 !important;
  overflow: hidden !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 0.35rem !important;
  visibility: visible !important;
  opacity: 1 !important;
}
/* Gradio wraps children — force the wrapper to a 2-row flex stack */
#col-center > .wrap,
#col-center > .form,
#col-center > div,
#col-right > .wrap,
#col-right > .form,
#col-right > div {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 0 !important;
  height: 100% !important;
  min-height: 0 !important;
  max-height: 100% !important;
  gap: 0.35rem !important;
  overflow: hidden !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#col-center .vp-cell,
#col-right .vp-cell {
  flex: 1 1 0 !important;
  min-height: 0 !important;
  height: auto !important;
  max-height: none !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
  padding: 0.25rem 0.35rem !important;
  box-sizing: border-box !important;
  scrollbar-width: none !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#col-center .vp-cell::-webkit-scrollbar,
#col-right .vp-cell::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}
/*
 * Hide only header chevron / scroll chrome — never hide .padded or
 * generic wrappers (those hold titles + Image content).
 */
#col-center .vp-cell > .wrap > button.icon-button,
#col-right .vp-cell > .wrap > button.icon-button,
#col-center .vp-cell > button.icon-button,
#col-right .vp-cell > button.icon-button,
#col-center .vp-cell .icon-button-wrapper:not(:has(img)),
#col-right .vp-cell .icon-button-wrapper:not(:has(img)) {
  display: none !important;
}
#col-center .vp-cell .viewport-title,
#col-right .vp-cell .viewport-title {
  flex: 0 0 auto !important;
  margin: 0 0 0.15rem 0 !important;
}
/* Plot fills remaining cell; images must stay visible */
#col-center .vp-cell .vp-plot,
#col-right .vp-cell .vp-plot,
#col-center .vp-cell [data-testid="image"],
#col-right .vp-cell [data-testid="image"] {
  flex: 1 1 0 !important;
  min-height: 0 !important;
  width: 100% !important;
  height: 100% !important;
  max-height: 100% !important;
  overflow: hidden !important;
  margin: 0 !important;
  border-radius: 8px !important;
  background: rgba(7, 11, 20, 0.35) !important;
  display: flex !important;
  flex-direction: column !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#col-center .vp-cell .image-container,
#col-right .vp-cell .image-container,
#col-center .vp-cell .vp-plot > div,
#col-right .vp-cell .vp-plot > div,
#col-center .vp-cell .vp-plot > .wrap,
#col-right .vp-cell .vp-plot > .wrap {
  flex: 1 1 0 !important;
  min-height: 0 !important;
  width: 100% !important;
  height: 100% !important;
  max-height: 100% !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: hidden !important;
  visibility: visible !important;
  opacity: 1 !important;
}
#col-center .vp-cell img,
#col-right .vp-cell img {
  max-width: 100% !important;
  max-height: 100% !important;
  width: 100% !important;
  height: 100% !important;
  object-fit: contain !important;
  object-position: center !important;
  display: block !important;
  margin: 0 auto !important;
  visibility: visible !important;
  opacity: 1 !important;
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
<div class="layer-panel" style="color:#cbd5e1;font-size:0.78rem;line-height:1.45;height:100%;overflow:auto;">
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


def _display_image(value, *, height="100%", elem_classes=None, **extra):
    """gr.Image for display-only plots — Gradio 4/5/6 keyword-compatible.

    Default height 100% so CSS flex-fit can size plots to the cell
    (fixed 42vh overflowed HF iframe chrome).
    """
    import inspect

    params = inspect.signature(gr.Image.__init__).parameters
    kwargs: dict = {
        "value": value,
        "label": None,
        "show_label": False,
        "interactive": False,
    }
    if height is not None:
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
    kwargs.update({k: v for k, v in extra.items() if k in params or k in ("visible",)})
    # Drop keys Image does not accept
    kwargs = {k: v for k, v in kwargs.items() if k in params or k == "elem_classes"}
    # elem_classes might be elem_classes in all versions
    if "elem_classes" not in params and "elem_classes" in kwargs:
        kwargs.pop("elem_classes")
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
    """
    Live replot of all three scan viewports (shell / radial / path).

    Always returns RGB stills at the same (frac, plane) so a plane change
    cancels any playing GIFs and keeps the suite synchronized.
    """
    return replot_scan_suite(
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


def _startup_plots() -> dict:
    """Precompute stub plots so the four viewports are never empty on first paint."""
    try:
        return run_pipeline(
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
    except Exception:
        logger.exception("startup plots failed")
        b = blank_rgb(420, 360)
        return {
            "shell": None,
            "img_shell": b,
            "img_radial": b,
            "img_field": b,
            "img_path": blank_rgb(320, 400),
            "img_metrics": blank_rgb(300, 400),
            "img_trace": b,
            "status_md": "### Seed status\n_Startup failed — click **Build**._",
        }


def build_app() -> gr.Blocks:
    # Gradio 5+: css/js/head are set on launch (and on Blocks when supported).
    # Slider fill JS MUST run on page load — head <script> is most reliable on HF.
    blocks_kwargs = dict(
        title="flux_trajectoid",
        analytics_enabled=False,
    )
    optional_block = {
        "css": CUSTOM_CSS,
        "js": SLIDER_FILL_JS,
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

        # ---- Workspace (single page, multi-column) ----
        with gr.Row(elem_id="workspace", equal_height=True):
            # Column 1 — CONTROLS | REFERENCES tabs
            with gr.Column(
                scale=5,
                min_width=300,
                elem_classes=["layer-inner"],
                elem_id="controls",
            ):
                with gr.Tabs(elem_id="col1-tabs", selected="controls") as col1_tabs:
                    with gr.Tab("CONTROLS", id="controls"):
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
                            elem_classes=["layer-fg"],
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
                                elem_classes=["layer-fg"],
                            )

            # Column 2 (~50%): top/bottom half viewports
            #   [ shell  ]
            #   [ radial ]
            with gr.Column(
                scale=9,
                min_width=0,
                elem_id="col-center",
                elem_classes=["vp-col"],
            ):
                with gr.Column(
                    min_width=0,
                    elem_classes=["layer-inner", "vp-cell"],
                    elem_id="vp-shell",
                ):
                    gr.Markdown(
                        '<p class="viewport-title">3D shell · contact path · matrix slice</p>'
                    )
                    img_shell = _display_image(
                        boot["img_shell"],
                        height=300,
                        elem_classes=["vp-plot"],
                    )
                with gr.Column(
                    min_width=0,
                    elem_classes=["layer-inner", "vp-cell"],
                    elem_id="vp-radial",
                ):
                    gr.Markdown(
                        '<p class="viewport-title">Radial trench / shave</p>'
                    )
                    img_radial = _display_image(
                        boot["img_radial"],
                        height=300,
                        elem_classes=["vp-plot"],
                    )

            # Column 3 (~25%): top/bottom half viewports (same heights as col-2)
            #   [ path      ]
            #   [ scorecard ]
            with gr.Column(
                scale=4,
                min_width=0,
                elem_id="col-right",
                elem_classes=["vp-col"],
            ):
                with gr.Column(
                    min_width=0,
                    elem_classes=["layer-inner", "vp-cell"],
                    elem_id="vp-path",
                ):
                    gr.Markdown(
                        '<p class="viewport-title">Rolling path (Nature-style)</p>'
                    )
                    img_path = _display_image(
                        boot["img_path"],
                        height=300,
                        elem_classes=["vp-plot"],
                    )
                with gr.Column(
                    min_width=0,
                    elem_classes=["layer-inner", "vp-cell"],
                    elem_id="vp-score",
                ):
                    gr.Markdown('<p class="viewport-title">Scorecard</p>')
                    img_metrics = _display_image(
                        boot["img_metrics"],
                        height=300,
                        elem_classes=["vp-plot"],
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
            """X/Y/Z → synced stills; XYZ → play X then Y then Z sequence."""
            pl = str(plane or "z").lower().strip()
            if pl == "xyz":
                return play_scan_ui(shell, "xyz", n_frames, ping_pong)
            s, r, p = update_slice_ui(shell, slice_frac, plane, show_slice)
            return s, r, p, gr.update()  # keep status

        def _plane_btn_prelude(plane):
            # Glow Play only while XYZ sequence is generating
            if str(plane or "").lower().strip() == "xyz":
                return _scan_btn_on()
            return gr.update()

        def _plane_btn_epilogue(plane):
            if str(plane or "").lower().strip() == "xyz":
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
  /*
   * Fit the host screen (e.g. 1920×1080) on HF Spaces.
   *
   * Bug we hit: body was forced TALLER than the short hub iframe with
   * overflow:hidden → UI clipped mid-panel and a huge black band below
   * the iframe. Fix:
   *  1) Target height ≈ parent/screen minus HF chrome
   *  2) Grow the hub iframe (parentIFrame + iframe-resizer messages)
   *  3) Pin every layout node to that height and split plot cells 50/50
   *  4) Never max-height clip below the target (use min-height + height)
   */
  function hostChrome() {
    // Spaces header + title row + App/Files tabs (approx)
    return 155;
  }

  function targetAppHeight() {
    // Prefer the *current* iframe viewport once it has grown; otherwise
    // request the host screen size so iFrameResizer/CSS grow expands us.
    const ih = window.innerHeight || document.documentElement.clientHeight || 0;
    const sh = (window.screen && (window.screen.availHeight || window.screen.height)) || 0;
    const screenTarget = sh > 0 ? sh - hostChrome() : 0;

    let h = ih;
    try {
      if (window.parent && window.parent !== window) {
        const ph = window.parent.innerHeight || 0;
        if (ph > 0) h = Math.max(h, ph - hostChrome());
      }
    } catch (_) { /* cross-origin hub page */ }

    // If iframe is still the short content-sized band, push toward screen
    if (screenTarget > 0 && (h < screenTarget * 0.85 || h < 560)) {
      h = screenTarget;
    }
    if (!h || h < 480) h = Math.max(480, screenTarget || 720);
    if (sh > 0) h = Math.min(h, sh - 60);
    return Math.round(h);
  }

  function requestIframeHeight(h) {
    try {
      if (window.parentIFrame) {
        try { window.parentIFrame.autoResize(true); } catch (_) {}
        try { window.parentIFrame.size(h); } catch (_) {}
      }
    } catch (_) {}
    // Gradio / iframe-resizer message formats
    const msgs = [
      '[iFrameSizer]height:' + h,
      '[iFrameSizer]height:' + h + ':force',
      { type: 'SET_IFRAME_HEIGHT', height: h },
      { type: 'iframe-height', height: h },
      { type: 'setHeight', height: h },
    ];
    msgs.forEach((m) => {
      try { window.parent && window.parent.postMessage(m, '*'); } catch (_) {}
    });
  }

  function ensureSpacer(h) {
    // In-flow element so document/content height = target (grows hub iframe)
    let s = document.getElementById('ft-spacer');
    if (!s) {
      s = document.createElement('div');
      s.id = 'ft-spacer';
      s.setAttribute('data-iframe-height', '');
      const host = document.body || document.documentElement;
      host.insertBefore(s, host.firstChild);
    }
    s.style.cssText = [
      'display:block',
      'width:1px',
      'height:' + h + 'px',
      'min-height:' + h + 'px',
      'margin:0',
      'padding:0',
      'pointer-events:none',
      'opacity:0',
      'overflow:hidden',
    ].join(';');
  }

  function fitScreen() {
    const h = targetAppHeight();
    const root = document.documentElement;
    root.style.setProperty('--ft-app-h', h + 'px');
    root.style.setProperty('height', h + 'px');
    root.style.setProperty('min-height', h + 'px');
    root.style.overflow = 'hidden';
    if (document.body) {
      document.body.style.setProperty('height', h + 'px');
      document.body.style.setProperty('min-height', h + 'px');
      document.body.style.overflow = 'hidden';
      document.body.style.margin = '0';
      document.body.style.padding = '0';
      document.body.style.background = '#070b14';
    }

    ensureSpacer(h);

    const nav = document.querySelector('#nav-bar');
    const navH = (nav && nav.offsetHeight) ? nav.offsetHeight : 48;
    root.style.setProperty('--ft-nav-h', navH + 'px');
    if (nav) {
      nav.style.setProperty('position', 'fixed', 'important');
      nav.style.setProperty('top', '0', 'important');
      nav.style.setProperty('left', '0', 'important');
      nav.style.setProperty('right', '0', 'important');
      nav.style.setProperty('height', navH + 'px', 'important');
      nav.style.setProperty('z-index', '1000', 'important');
    }

    // Fixed workspace fills the iframe viewport under the nav
    const ws = document.querySelector('#workspace');
    if (ws) {
      ws.style.setProperty('position', 'fixed', 'important');
      ws.style.setProperty('top', navH + 'px', 'important');
      ws.style.setProperty('left', '0', 'important');
      ws.style.setProperty('right', '0', 'important');
      ws.style.setProperty('bottom', '0', 'important');
      ws.style.setProperty('height', 'auto', 'important');
      ws.style.setProperty('width', '100%', 'important');
      ws.style.setProperty('display', 'grid', 'important');
      ws.style.setProperty('overflow', 'hidden', 'important');
      ws.style.setProperty('z-index', '900', 'important');
    }

    // Columns stretch to the fixed workspace box
    const wh = Math.max(200, (window.innerHeight || h) - navH);
    ['#controls', '#col-center', '#col-right'].forEach((sel) => {
      const el = document.querySelector(sel);
      if (!el) return;
      el.style.setProperty('height', '100%', 'important');
      el.style.setProperty('min-height', '0', 'important');
      el.style.setProperty('max-height', '100%', 'important');
      el.style.setProperty('overflow', sel === '#controls' ? 'auto' : 'hidden', 'important');
      el.style.setProperty('min-width', '0', 'important');
      el.style.setProperty('visibility', 'visible', 'important');
      el.style.setProperty('opacity', '1', 'important');
      if (sel !== '#controls') {
        el.style.setProperty('display', 'flex', 'important');
        el.style.setProperty('flex-direction', 'column', 'important');
      } else {
        el.style.setProperty('display', 'flex', 'important');
        el.style.setProperty('flex-direction', 'column', 'important');
      }
    });

    // Ensure CONTROLS tab body is visible (Gradio manages inactive panels)
    const top = document.querySelector('#controls-top');
    if (top) {
      top.style.setProperty('display', 'flex', 'important');
      top.style.setProperty('flex-direction', 'column', 'important');
      top.style.setProperty('visibility', 'visible', 'important');
      top.style.setProperty('opacity', '1', 'important');
      top.style.setProperty('height', 'auto', 'important');
      top.style.setProperty('max-height', 'none', 'important');
      top.style.setProperty('overflow', 'visible', 'important');
    }
    // If every tabitem is display:none (bad CSS residue), show the first
    const panels = Array.from(
      document.querySelectorAll('#col1-tabs .tabitem, #col1-tabs [role="tabpanel"]')
    );
    if (panels.length && panels.every((p) => {
      const d = (p.style && p.style.display) || getComputedStyle(p).display;
      return d === 'none';
    })) {
      const first = panels[0];
      first.style.setProperty('display', 'block', 'important');
      first.style.setProperty('visibility', 'visible', 'important');
      first.style.setProperty('height', 'auto', 'important');
      first.style.setProperty('overflow', 'auto', 'important');
    }

    // Plot columns: force two equal rows (shell/radial · path/scorecard)
    ['#col-center', '#col-right'].forEach((sel) => {
      const col = document.querySelector(sel);
      if (!col) return;
      col.style.setProperty('display', 'flex', 'important');
      col.style.setProperty('flex-direction', 'column', 'important');
      col.style.setProperty('height', wh + 'px', 'important');
      col.style.setProperty('min-height', wh + 'px', 'important');
      col.style.setProperty('overflow', 'hidden', 'important');
      // Gradio wrapper holding both vp-cells
      col.querySelectorAll(':scope > div, :scope > .wrap, :scope > .form').forEach((w) => {
        w.style.setProperty('display', 'flex', 'important');
        w.style.setProperty('flex-direction', 'column', 'important');
        w.style.setProperty('flex', '1 1 0', 'important');
        w.style.setProperty('height', '100%', 'important');
        w.style.setProperty('min-height', '0', 'important');
        w.style.setProperty('gap', '0.35rem', 'important');
        w.style.setProperty('overflow', 'hidden', 'important');
      });
      col.querySelectorAll('.vp-cell').forEach((cell) => {
        cell.style.setProperty('flex', '1 1 0', 'important');
        cell.style.setProperty('min-height', '0', 'important');
        cell.style.setProperty('height', 'auto', 'important');
        cell.style.setProperty('max-height', 'none', 'important');
        cell.style.setProperty('display', 'flex', 'important');
        cell.style.setProperty('flex-direction', 'column', 'important');
        cell.style.setProperty('overflow', 'hidden', 'important');
        cell.style.setProperty('visibility', 'visible', 'important');
        cell.style.setProperty('opacity', '1', 'important');
      });
    });

    document.querySelectorAll(
      '#col-center .vp-cell img, #col-right .vp-cell img, #col-center .vp-plot, #col-right .vp-plot'
    ).forEach((el) => {
      el.style.setProperty('max-height', '100%', 'important');
      el.style.setProperty('max-width', '100%', 'important');
      el.style.setProperty('width', '100%', 'important');
      el.style.setProperty('height', '100%', 'important');
      el.style.setProperty('object-fit', 'contain', 'important');
      el.style.setProperty('visibility', 'visible', 'important');
      el.style.setProperty('opacity', '1', 'important');
      el.style.setProperty('display', el.tagName === 'IMG' ? 'block' : 'flex', 'important');
    });

    requestIframeHeight(h);

    // If the hub iframe grew, re-fit cells to the new innerHeight
    const ih = window.innerHeight || h;
    if (Math.abs(ih - h) > 40) {
      root.style.setProperty('--ft-app-h', ih + 'px');
    }
  }

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

  /* Thin analog line (~1.5px) + mild glow — never full input height */
  const FILL_STYLE = [
    'position:absolute',
    'left:0',
    'top:50%',
    'transform:translateY(-50%)',
    'height:1.5px',
    'min-height:1.5px',
    'max-height:1.5px',
    'border-radius:999px',
    'background:#00FF00',
    'background-color:#00FF00',
    'box-shadow:0 0 2px 0.4px rgba(0,255,0,0.7),0 0 4px 0.8px rgba(0,255,0,0.35)',
    'pointer-events:none',
    'z-index:2',
    'display:block',
    'opacity:1',
    'visibility:visible',
    'margin:0',
    'padding:0',
    'border:none',
  ].join(';');

  const RAIL_STYLE = [
    'position:absolute',
    'left:0',
    'right:0',
    'top:50%',
    'transform:translateY(-50%)',
    'height:1.5px',
    'border-radius:999px',
    'background:rgba(100,116,139,0.45)',
    'pointer-events:none',
    'z-index:1',
  ].join(';');

  function ensureShell(el) {
    let shell = el.closest('.ft-slider-shell');
    if (!shell) {
      shell = document.createElement('div');
      shell.className = 'ft-slider-shell';
      shell.style.cssText = [
        'position:relative',
        'display:block',
        'width:100%',
        'height:14px',
        'margin:0.3rem 0 0.5rem 0',
        'overflow:visible',
        'box-sizing:border-box',
      ].join(';');
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
    // keep fill above rail, under input hit-target
    if (fill.nextSibling !== el) {
      shell.insertBefore(fill, el);
    }
    return shell;
  }

  function paint(el) {
    if (!el || el.type !== 'range') return;
    const shell = ensureShell(el);
    if (!shell) return;
    const pct = pctOf(el);
    const fill = shell.querySelector(':scope > .ft-slider-fill');
    if (!fill) return;

    // Width in px to the knob center (accounts for thumb radius)
    const shellW = shell.clientWidth || shell.getBoundingClientRect().width || 0;
    const thumb = 12;
    let wPx;
    if (shellW > 0) {
      // value maps across (width - thumb) with thumb centered
      wPx = Math.max(0, (thumb / 2) + ((shellW - thumb) * pct) / 100);
    } else {
      wPx = null;
    }

    fill.style.cssText = FILL_STYLE + ';width:' + (
      wPx != null ? (wPx.toFixed(2) + 'px') : (pct.toFixed(3) + '%')
    );

    const pctStr = pct.toFixed(3) + '%';
    shell.style.setProperty('--ft-fill-pct', pctStr);
    el.style.setProperty('--ft-fill-pct', pctStr);
    // Keep input body fully transparent — fat green bar was from full-height bg
    el.style.setProperty('background', 'transparent', 'important');
    el.style.setProperty('background-image', 'none', 'important');
    el.style.setProperty('background-color', 'transparent', 'important');
  }

  function bindAll() {
    document.querySelectorAll('input[type="range"]').forEach((el) => {
      if (!el.closest('.ft-slider-shell')) el.dataset.ftSliderBound = '0';
      if (el.dataset.ftSliderBound === '1') {
        paint(el);
        return;
      }
      el.dataset.ftSliderBound = '1';
      ensureShell(el);
      const upd = () => paint(el);
      ['input', 'change', 'pointerdown', 'pointerup', 'touchstart', 'touchmove'].forEach((ev) => {
        el.addEventListener(ev, upd, { passive: true });
      });
      el.addEventListener('pointermove', () => {
        if (el.matches(':active')) paint(el);
      }, { passive: true });
      paint(el);
    });
  }
  // Gradio may inject this script before components mount
  let fitTimer = null;
  let paintTimer = null;
  function scheduleFit() {
    if (fitTimer) return;
    fitTimer = setTimeout(() => {
      fitTimer = null;
      fitScreen();
      bindNavTabs();
    }, 50);
  }
  function schedulePaint() {
    if (paintTimer) return;
    paintTimer = setTimeout(() => {
      paintTimer = null;
      bindAll();
    }, 30);
  }
  const start = () => {
    fitScreen();
    bindAll();
    bindNavTabs();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
  window.addEventListener('resize', scheduleFit, { passive: true });
  window.addEventListener('orientationchange', scheduleFit, { passive: true });
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', scheduleFit, { passive: true });
  }
  window.addEventListener('message', (ev) => {
    if (!ev || ev.data == null) return;
    const d = ev.data;
    if (d === 'fit' || d?.type === 'resize' || d?.type === 'SET_SCROLLING') {
      scheduleFit();
    }
  });
  // Any DOM change: re-shell / re-paint sliders (Gradio recreates ranges)
  const obs = new MutationObserver(() => {
    schedulePaint();
    scheduleFit();
  });
  obs.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['value', 'style', 'class'],
  });
  // Continuous paint while app is live (covers Gradio value writes)
  setInterval(bindAll, 400);
  let n = 0;
  const boot = setInterval(() => {
    fitScreen();
    bindAll();
    n += 1;
    if (n > 40) clearInterval(boot);
  }, 500);
  setTimeout(start, 0);
  setTimeout(start, 200);
  setTimeout(start, 600);
  setTimeout(start, 1500);
  setTimeout(start, 3000);
})();
"""


def _launch(demo, *, port: int, theme, css: str, js: str) -> None:
    """Launch with kwargs filtered to the installed Gradio version."""
    import inspect

    # Ensure Blocks carries css/js even if constructor ignored them
    for attr, val in (("css", css), ("js", js)):
        try:
            setattr(demo, attr, val)
        except Exception:
            pass
    try:
        if getattr(demo, "head", None) in (None, ""):
            demo.head = f"<script>\n{js}\n</script>"
    except Exception:
        pass

    queued = demo.queue(default_concurrency_limit=1)
    base = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "share": False,
        "show_error": True,
    }
    optional = {
        "theme": theme,
        "css": css,
        "js": js,
        "head": f"<script>\n{js}\n</script>",
        "ssr": False,
        "ssr_mode": False,
        "show_api": False,
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
