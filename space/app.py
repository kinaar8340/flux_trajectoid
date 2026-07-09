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

/* Top navigation */
#nav-bar {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.75rem;
  background: rgba(15, 23, 42, 0.92);
  border-bottom: 1px solid rgba(56, 189, 248, 0.25);
  min-height: 44px;
  max-height: 48px;
}
#nav-bar .nav-title {
  color: #e2e8f0;
  font-weight: 600;
  font-size: 0.95rem;
  margin-right: 0.75rem;
  letter-spacing: 0.02em;
}
#nav-bar .nav-links a, #nav-bar button {
  color: #94a3b8 !important;
  font-size: 0.78rem !important;
  padding: 0.25rem 0.55rem !important;
  border-radius: 6px !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  min-width: auto !important;
}
#nav-bar button:hover {
  color: #e0f2fe !important;
  border-color: rgba(56, 189, 248, 0.35) !important;
  background: rgba(56, 189, 248, 0.08) !important;
}
#nav-bar button.primary, #nav-bar .nav-active {
  color: #0f172a !important;
  background: #38bdf8 !important;
  border-color: #38bdf8 !important;
}
#nav-meta {
  margin-left: auto;
  color: #64748b;
  font-size: 0.72rem;
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

/* Independent vertical scrollbars on all three columns */
#controls,
#col-center,
#col-right {
  height: 100% !important;
  max-height: 100% !important;
  min-height: 0 !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  scrollbar-width: thin;
  scrollbar-color: rgba(56, 189, 248, 0.45) rgba(15, 23, 42, 0.4);
}
#controls::-webkit-scrollbar,
#col-center::-webkit-scrollbar,
#col-right::-webkit-scrollbar {
  width: 8px;
}
#controls::-webkit-scrollbar-thumb,
#col-center::-webkit-scrollbar-thumb,
#col-right::-webkit-scrollbar-thumb {
  background: rgba(56, 189, 248, 0.4);
  border-radius: 4px;
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
            use_tpt=bool(use_tpt),
            build_3d=bool(build_3d),
            force_stub=bool(force_stub),
            slice_frac=float(slice_frac),
            slice_plane=str(slice_plane or "z"),
            show_slice=bool(show_slice),
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
        show_slice=bool(show_slice),
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
        ping_pong=bool(ping_pong),
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
        title="flux_trajectoid · Photon Seed Asteroids",
        analytics_enabled=False,
    ) as demo:
        # ---- Top navigation ----
        with gr.Row(elem_id="nav-bar"):
            gr.HTML(
                '<span class="nav-title">✦ flux_trajectoid</span>'
                '<span style="color:#475569;font-size:0.72rem;">Photon Seed Asteroids</span>'
            )
            btn_demo = gr.Button("Demo", size="sm", elem_classes=["nav-active"])
            btn_ref = gr.Button("Reference", size="sm")
            btn_about = gr.Button("About", size="sm")
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
            # Column 1 — compact collapsed accordions + Seed Status fills rest
            with gr.Column(
                scale=2, min_width=200, elem_classes=["layer-inner"], elem_id="controls"
            ):
                with gr.Column(elem_id="controls-top"):
                    gr.Markdown('<p class="viewport-title">Controls</p>')
                    # Startup: all four accordions collapsed
                    with gr.Accordion(
                        "Payload & identity", open=False, elem_classes=["layer-fg"]
                    ):
                        payload = gr.Textbox(
                            value="Hello from the shell",
                            label="Payload",
                            lines=1,
                            max_lines=1,
                        )
                        seed = gr.Number(value=42, label="Seed", precision=0)
                    with gr.Accordion("Channel", open=False, elem_classes=["layer-fg"]):
                        turbulence = gr.Slider(
                            0.0, 0.8, value=0.25, step=0.05, label="Turbulence"
                        )
                        n_steps = gr.Slider(
                            4, 32, value=12, step=1, label="Channel steps"
                        )
                    with gr.Accordion(
                        "Shell / lattice", open=False, elem_classes=["layer-fg"]
                    ):
                        use_tpt = gr.Checkbox(value=True, label="TPT closure")
                        build_3d = gr.Checkbox(value=True, label="3D shell")
                        force_stub = gr.Checkbox(
                            value=True, label="Fast stub lattice"
                        )
                    with gr.Accordion(
                        "Matrix slice", open=False, elem_classes=["layer-fg"]
                    ):
                        show_slice = gr.Checkbox(value=True, label="Show green slice")
                        slice_plane = gr.Radio(
                            choices=["x", "y", "z"],
                            value="z",
                            label="Slice plane",
                            type="value",
                        )
                        slice_frac = gr.Slider(
                            0.0,
                            1.0,
                            value=0.5,
                            step=0.02,
                            label="Slice position",
                        )
                        scan_frames = gr.Slider(
                            8,
                            24,
                            value=14,
                            step=1,
                            label="Scan frames",
                        )
                        scan_ping = gr.Checkbox(
                            value=False,
                            label="Ping-pong scan (can look bouncy)",
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

            # Column 2 — primary viewports (scrollable)
            with gr.Column(
                scale=4, min_width=320, elem_classes=["layer-inner"], elem_id="col-center"
            ):
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown(
                            '<p class="viewport-title">3D shell · contact path · matrix slice</p>'
                        )
                        img_shell = gr.Image(
                            value=blank_rgb(300, 360),
                            label=None,
                            show_label=False,
                            height=260,
                        )
                    with gr.Column():
                        gr.Markdown('<p class="viewport-title">Radial trench / shave</p>')
                        img_radial = gr.Image(
                            value=blank_rgb(300, 360),
                            label=None,
                            show_label=False,
                            height=260,
                        )
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown('<p class="viewport-title">Protected OAM field</p>')
                        img_field = gr.Image(
                            value=blank_rgb(260, 360),
                            label=None,
                            show_label=False,
                            height=220,
                        )
                    with gr.Column():
                        gr.Markdown('<p class="viewport-title">Fidelity trace</p>')
                        img_trace = gr.Image(
                            value=blank_rgb(200, 360),
                            label=None,
                            show_label=False,
                            height=220,
                        )

            # Column 3 — path + metrics + reference (scrollable)
            with gr.Column(
                scale=3, min_width=240, elem_classes=["layer-inner"], elem_id="col-right"
            ):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        gr.Markdown('<p class="viewport-title">Rolling path (Nature-style)</p>')
                        img_path = gr.Image(
                            value=blank_rgb(400, 220),
                            label=None,
                            show_label=False,
                            height=300,
                        )
                    with gr.Column(scale=3):
                        gr.Markdown('<p class="viewport-title">Scorecard</p>')
                        img_metrics = gr.Image(
                            value=blank_rgb(260, 360),
                            label=None,
                            show_label=False,
                            height=160,
                        )
                        gr.Markdown('<p class="viewport-title">Reference · trajectoid gallery</p>')
                        hero_img = gr.Image(
                            value=hero,
                            label=None,
                            show_label=False,
                            height=160,
                            elem_classes=["layer-fg"],
                        )
                        gr.Markdown(
                            '<p id="hero-caption">Trajectoid shapes + theory/experiment paths · used as HF Space visual language</p>',
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
            # Swap center hero: show construction figure in shell viewport caption via status
            ref = asset_path("shell_construction.png")
            hero_p = asset_path("trajectoid_paths.png")
            md = (
                "### Reference figures\n"
                "**(top)** Trajectoid polyhedra + rolling paths (theory black / experiment blue).\n\n"
                "**(construction)** Inclined path T, potential surface, shell/core/remnant "
                "with shaved region — the geometric language of Photon Seed Asteroids.\n"
            )
            # Keep plots; update hero + status
            return (
                shell_img,
                radial_img,
                field_img,
                path_img,
                metrics_img,
                trace_img,
                md,
                gr.update(value=ref or hero_p),
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
            )

        btn_ref.click(
            fn=show_ref,
            inputs=outs,
            outputs=outs + [hero_img],
        )
        btn_about.click(
            fn=show_about,
            inputs=outs,
            outputs=outs + [hero_img],
        )
        btn_demo.click(
            fn=lambda *a: a[-1],  # no-op keep
            inputs=outs,
            outputs=[status],
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
        )
    except TypeError:
        # Fallback: inject CSS via head if launch rejects css=
        demo.css = CUSTOM_CSS  # type: ignore[attr-defined]
        queued.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
        )
