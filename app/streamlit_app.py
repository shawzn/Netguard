from __future__ import annotations

import base64
import html
import json
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from netguard.predict import NetGuardPredictor  # noqa: E402

# Balanced 15x9 pixel bitmap for geometric bold 'N' and 'G'
NG_PIXEL_BITMAP = [
    "110001100111110",
    "111001101111111",
    "111101101100000",
    "111101101100000",
    "110111101100111",
    "110111101100111",
    "110011101100011",
    "110001101111111",
    "110001100111110",
]


def _ng_pixel_rects(
    offset_x: float, offset_y: float, step: float, size: float, fill: str
) -> str:
    rects: list[str] = []
    for r_idx, row in enumerate(NG_PIXEL_BITMAP):
        for c_idx, cell in enumerate(row):
            if cell == "1":
                x = offset_x + c_idx * step
                y = offset_y + r_idx * step
                rects.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" '
                    f'height="{size:.2f}" fill="{fill}"/>'
                )
    return "".join(rects)


def build_ng_pixel_favicon() -> str:
    """Square double-bordered offset badge with pixel 'NG' for browser tab icon."""
    rects = _ng_pixel_rects(
        offset_x=13.5, offset_y=20.0, step=2.2, size=2.05, fill="#080808"
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect x="11" y="11" width="46" height="46" fill="#080808" stroke="#9ca3af" stroke-width="2"/>'
        '<rect x="6" y="6" width="46" height="46" fill="#f5f5f5" stroke="#080808" stroke-width="3"/>'
        f"{rects}"
        "</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def build_ng_pixel_logo_svg(fill: str = "#080808", step: float = 1.48, size: float = 1.38) -> str:
    """Inline SVG pixel 'NG' centered inside the square offset badge."""
    rects = _ng_pixel_rects(offset_x=0.0, offset_y=0.0, step=step, size=size, fill=fill)
    w = 15 * step
    h = 9 * step
    return (
        f'<svg width="{w:.2f}" height="{h:.2f}" viewBox="0 0 {w:.2f} {h:.2f}" '
        f'fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f"{rects}</svg>"
    )


st.set_page_config(
    page_title="NetGuard — Network Intrusion Detection",
    page_icon=build_ng_pixel_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAV_ITEMS = [
    "Overview Dashboard",
    "Dataset Analysis",
    "Model Performance",
    "Intrusion Scanner",
]

NAV_SUBTITLES = {
    "Overview Dashboard": "Session detection telemetry, active model state, and scan ledger.",
    "Dataset Analysis": "Distributions, high-correlation pairs, and SMOTENC resampling partitions on NSL-KDD.",
    "Model Performance": "Controlled external evaluation of XGBoost Baseline vs. XGBoost + SMOTENC on KDDTest+.",
    "Intrusion Scanner": "Offline batch inference for uploaded NSL-KDD CSV captures (up to 20 records).",
}


def inject_theme_and_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

        :root {
            --bg: #080808;
            --panel: rgba(12, 12, 12, 0.78);
            --panel-solid: #0e0e0e;
            --tx: #e4e4e7;
            --tx-bright: #fafafa;
            --sub: #94949c;
            --bd: rgba(255, 255, 255, 0.11);
            --bd-hover: rgba(255, 255, 255, 0.26);
            --ok: #4ade80;
            --warn: #f5b942;
            --crit: #ff6161;
            --ng-nav-h: 64px;
        }

        /* Cross-platform normalization (Android, iOS Safari, macOS, Windows, Linux) */
        *, *::before, *::after {
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }

        html {
            -webkit-text-size-adjust: 100%;
            scroll-behavior: smooth;
            background-color: #080808 !important;
            overflow-x: clip !important;
        }

        body {
            background-color: #080808 !important;
            color: var(--tx) !important;
            font: 14px/1.55 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow-x: clip !important;
            min-height: 100dvh;
        }

        /* Keep Streamlit wrappers transparent so #ng-dither-bg-canvas renders cleanly underneath */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section.main {
            background: transparent !important;
            background-color: transparent !important;
            color: var(--tx) !important;
            position: relative;
            z-index: 1;
            padding: 0 !important;
            margin: 0 !important;
            min-height: 100dvh !important;
        }

        /* Hide Streamlit native chrome */
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        #MainMenu,
        footer[data-testid="stFooter"],
        .stApp > footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        /* Collapse zero-height JS bridge container */
        div[data-testid="stElementContainer"]:has(iframe[height="0"]),
        div[data-testid="stVerticalBlock"] > div:has(iframe[height="0"]) {
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Main container: dynamic navbar clearance + zero trailing gap */
        [data-testid="stMainBlockContainer"],
        .block-container {
            max-width: 1140px !important;
            width: 100% !important;
            min-height: 100dvh !important;
            margin: 0 auto !important;
            padding-top: calc(var(--ng-nav-h, 64px) + 24px) !important;
            padding-bottom: 0 !important;
            padding-left: max(20px, env(safe-area-inset-left, 20px)) !important;
            padding-right: max(20px, env(safe-area-inset-right, 20px)) !important;
            display: flex !important;
            flex-direction: column !important;
        }

        [data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
            flex: 1 0 auto !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 14px !important;
        }

        /* Push footer wrapper flush to bottom */
        [data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:has(footer.ng-footer) {
            margin-top: auto !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* =========================================================
           FIXED FULL-WIDTH NAVBAR (Pixel-centered logo & balanced tabs)
           ========================================================= */
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            z-index: 99999 !important;
            margin: 0 !important;
            padding-top: calc(10px + env(safe-area-inset-top, 0px)) !important;
            padding-bottom: 10px !important;
            padding-left: max(20px, calc((100% - 1140px) / 2 + 20px), env(safe-area-inset-left, 20px)) !important;
            padding-right: max(20px, calc((100% - 1140px) / 2 + 20px), env(safe-area-inset-right, 20px)) !important;
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 20px !important;
            backdrop-filter: blur(16px) saturate(130%) !important;
            -webkit-backdrop-filter: blur(16px) saturate(130%) !important;
            background: rgba(8, 8, 8, 0.86) !important;
            border-bottom: 1px solid var(--bd) !important;
        }

        /* Normalize all nested Streamlit wrappers inside navbar so nothing sinks */
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) [data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) [data-testid="stElementContainer"],
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) [data-testid="stMarkdown"],
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) [data-testid="stMarkdownContainer"],
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) [data-testid="stButton"] {
            margin: 0 !important;
            padding: 0 !important;
            gap: 0 !important;
            min-height: 0 !important;
            width: auto !important;
            display: flex !important;
            align-items: center !important;
            line-height: 1 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="stColumn"]:first-child,
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="column"]:first-child {
            flex: 1 1 auto !important;
            width: auto !important;
            min-width: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="stColumn"]:not(:first-child),
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="column"]:not(:first-child) {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: max-content !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
        }

        /* Brand Logo + Wordmark centered cleanly above navbar line */
        .ng-nav-anchor {
            display: inline-flex;
            align-items: center;
            gap: 11px;
            height: 36px;
            margin: 0;
            padding: 0;
        }

        .ng-pixel-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            flex-shrink: 0;
            background: #f5f5f5;
            border: 1.5px solid #080808;
            box-shadow: 2px 2px 0px #080808, 3px 3px 0px rgba(255, 255, 255, 0.45);
            transform: translate(-1px, -1.5px);
            user-select: none;
            transition: transform 0.14s ease, box-shadow 0.14s ease;
        }

        .ng-pixel-badge:hover {
            transform: translate(-2px, -2.5px);
            box-shadow: 3px 3px 0px #080808, 4px 4px 0px rgba(255, 255, 255, 0.72);
        }

        .ng-brand-text {
            display: inline-flex;
            align-items: baseline;
            gap: 7px;
            line-height: 1;
            user-select: none;
        }

        .ng-brand-name {
            font: 600 14px/1 'Geist', sans-serif;
            letter-spacing: -0.02em;
            color: var(--tx-bright);
        }

        .ng-brand-tag {
            font: 500 10px/1 'Geist Mono', monospace;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--sub);
            padding: 2px 6px;
            border: 1px solid var(--bd);
            border-radius: 4px;
        }

        /* Navbar Tab Buttons */
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            border-bottom: 1.5px solid transparent !important;
            box-shadow: none !important;
            outline: none !important;
            border-radius: 0 !important;
            padding: 6px 2px !important;
            min-height: 32px !important;
            height: 32px !important;
            width: auto !important;
            white-space: nowrap !important;
            font-family: 'Geist', sans-serif !important;
            font-size: 13px !important;
            letter-spacing: -0.01em !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: color 0.15s ease, border-color 0.15s ease !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button p {
            white-space: nowrap !important;
            margin: 0 !important;
            line-height: 1.1 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button[kind="secondary"] {
            color: var(--sub) !important;
            font-weight: 400 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button[kind="secondary"]:hover {
            color: var(--tx-bright) !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button[kind="primary"] {
            color: var(--tx-bright) !important;
            font-weight: 600 !important;
            border-bottom-color: var(--tx-bright) !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button:focus,
        div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button:active {
            background: transparent !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* =========================================================
           HERO & TYPOGRAPHY
           ========================================================= */
        .ng-mono {
            font: 11px/1.35 'Geist Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-variant-numeric: tabular-nums;
        }

        .ng-hero {
            padding: 4px 0 6px 0;
        }

        .ng-tag {
            color: var(--sub);
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .ng-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--ok);
            display: inline-block;
        }

        .ng-h1 {
            font: 600 clamp(28px, 4.6vw, 52px)/1.04 'Geist', sans-serif;
            letter-spacing: -0.035em;
            color: var(--tx-bright);
            margin: 10px 0 10px 0;
        }

        .ng-sub {
            color: var(--sub);
            font-size: 15px;
            max-width: 640px;
            margin: 0;
            line-height: 1.55;
        }

        /* =========================================================
           KPI STRIP (Symmetric grid & uniform card height)
           ========================================================= */
        .ng-kpi-grid {
            display: grid;
            gap: 12px;
            width: 100%;
            margin-bottom: 2px;
        }

        .ng-kpi-grid--4 {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .ng-kpi-grid--5 {
            grid-template-columns: repeat(5, minmax(0, 1fr));
        }

        .ng-kpi {
            background: var(--panel);
            border: 1px solid var(--bd);
            border-radius: 10px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 14px 15px;
            min-height: 98px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 6px;
            min-width: 0;
        }

        .ng-kpi .v {
            font: 600 clamp(20px, 2.2vw, 25px)/1.15 'Geist', sans-serif;
            letter-spacing: -0.03em;
            color: var(--tx-bright);
            font-variant-numeric: tabular-nums;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .ng-kpi .s {
            font-size: 12px;
            color: var(--sub);
            line-height: 1.35;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        /* =========================================================
           EQUAL-HEIGHT PANELS & CHARTS
           ========================================================= */
        div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) {
            align-items: stretch !important;
            gap: 12px !important;
        }

        div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) > div[data-testid="stColumn"] {
            display: flex !important;
            flex-direction: column !important;
            min-width: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:only-child,
        div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:only-child > div[data-testid="stMarkdown"],
        div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:only-child > div[data-testid="stMarkdown"] > div[data-testid="stMarkdownContainer"] {
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
        }

        .ng-panel {
            background: var(--panel);
            border: 1px solid var(--bd);
            border-radius: 10px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 15px 16px;
            height: 100%;
            width: 100%;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .ng-panel-body {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            min-width: 0;
        }

        .ng-panel h3 {
            font: 500 14px/1.3 'Geist', sans-serif;
            color: var(--tx-bright);
            margin: 0 0 12px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .ng-panel h3 span {
            font: 10.5px 'Geist Mono', monospace;
            color: var(--sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .ng-num {
            font: 11px 'Geist Mono', monospace;
            font-style: normal;
            color: var(--tx-bright);
            opacity: 0.72;
            margin-right: 8px;
        }

        .ng-empty {
            border: 1px dashed var(--bd);
            border-radius: 8px;
            padding: 24px 16px;
            text-align: center;
            color: var(--sub);
            margin: auto 0;
        }

        .ng-empty b {
            display: block;
            color: var(--tx-bright);
            font: 18px 'Geist Mono', monospace;
            margin-bottom: 4px;
        }

        /* Confusion Matrix inside panel */
        .ng-cm-wrap {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 2px;
        }

        .ng-cm-grid {
            display: grid;
            grid-template-columns: 78px repeat(2, minmax(0, 1fr));
            gap: 6px;
            align-items: stretch;
        }

        .ng-cm-head {
            font: 10.5px/1.2 'Geist Mono', monospace;
            color: var(--sub);
            text-transform: uppercase;
            text-align: center;
            padding: 4px 2px;
        }

        .ng-cm-rowlabel {
            font: 10.5px/1.25 'Geist Mono', monospace;
            color: var(--sub);
            text-transform: uppercase;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            text-align: right;
            padding-right: 6px;
        }

        .ng-cm-cell {
            border: 1px solid var(--bd);
            border-radius: 7px;
            padding: 14px 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 3px;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }

        .ng-cm-cell .k {
            font: 10.5px 'Geist Mono', monospace;
            letter-spacing: 0.06em;
            opacity: 0.75;
        }

        .ng-cm-cell .n {
            font: 600 20px/1.15 'Geist', sans-serif;
            letter-spacing: -0.02em;
        }

        .ng-cm-cell .p {
            font: 11px 'Geist Mono', monospace;
            opacity: 0.72;
        }

        /* Feature Importance bars inside panel */
        .ng-fi-row {
            display: grid;
            grid-template-columns: minmax(115px, 165px) 1fr 56px;
            align-items: center;
            gap: 10px;
            margin: 5px 0;
            font-size: 12px;
        }

        .ng-fi-name {
            font: 11.5px 'Geist Mono', monospace;
            color: var(--tx);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .ng-fi-track {
            height: 9px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 3px;
            overflow: hidden;
        }

        .ng-fi-fill {
            height: 100%;
            background: linear-gradient(90deg, #9ca3af 0%, #f5f5f5 100%);
            border-radius: 3px;
        }

        .ng-fi-val {
            font: 11px 'Geist Mono', monospace;
            color: var(--sub);
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        /* Tables */
        .ng-sc {
            overflow: auto;
            -webkit-overflow-scrolling: touch;
            max-height: 360px;
            width: 100%;
        }

        .ng-tbl {
            border-collapse: collapse;
            width: 100%;
            font-size: 12.5px;
            font-variant-numeric: tabular-nums;
        }

        .ng-tbl th {
            text-align: left;
            color: var(--sub);
            font: 500 11px/1.3 'Geist Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 8px 10px;
            border-bottom: 1px solid var(--bd);
            white-space: nowrap;
        }

        .ng-tbl td {
            padding: 7px 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            color: var(--tx);
            white-space: nowrap;
        }

        .ng-tbl tbody tr:last-child td {
            border-bottom: none;
        }

        /* Horizontal Bars */
        .ng-bar {
            display: grid;
            grid-template-columns: 76px 1fr auto;
            align-items: center;
            gap: 10px;
            margin: 8px 0;
            font-size: 12px;
        }

        .ng-bar-track {
            height: 11px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 3px;
            overflow: hidden;
        }

        .ng-bar-fill {
            height: 100%;
            border-radius: 3px;
        }

        /* Status Badges */
        .ng-pill {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 99px;
            border: 1px solid var(--bd);
            font: 10.5px/1.35 'Geist Mono', monospace;
            margin-right: 4px;
            white-space: nowrap;
        }

        .ng-pill.ok {
            color: var(--ok);
            border-color: rgba(74, 222, 128, 0.48);
            background: rgba(74, 222, 128, 0.08);
        }

        .ng-pill.warn {
            color: var(--warn);
            border-color: rgba(245, 185, 66, 0.48);
            background: rgba(245, 185, 66, 0.08);
        }

        .ng-pill.crit {
            color: var(--crit);
            border-color: rgba(255, 97, 97, 0.5);
            background: rgba(255, 97, 97, 0.09);
        }

        /* Scanner Prompt & Control Cards */
        .ng-ok-card {
            border: 1px solid var(--bd);
            border-radius: 10px;
            background: var(--panel);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 16px;
        }

        .ng-ok-card h4 {
            margin: 6px 0 4px 0;
            font: 600 17px/1.3 'Geist', sans-serif;
            color: var(--tx-bright);
        }

        /* Streamlit native inputs & buttons harmonization */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stCheckbox"] label {
            color: var(--tx) !important;
            font-size: 13px !important;
        }

        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stFileUploader"] section {
            background-color: var(--panel) !important;
            border: 1px solid var(--bd) !important;
            color: var(--tx) !important;
            border-radius: 10px !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
        }

        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span {
            color: var(--sub) !important;
        }

        section.main div[data-testid="stVerticalBlock"] > div:not(:has(.ng-nav-anchor)) button[kind="primary"] {
            background-color: #f5f5f5 !important;
            color: #080808 !important;
            border: 1px solid #f5f5f5 !important;
            border-radius: 8px !important;
            min-height: 40px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }

        section.main div[data-testid="stVerticalBlock"] > div:not(:has(.ng-nav-anchor)) button[kind="secondary"],
        div[data-testid="stDownloadButton"] button {
            background-color: var(--panel) !important;
            color: var(--tx) !important;
            border: 1px solid var(--bd) !important;
            border-radius: 8px !important;
            min-height: 40px !important;
            font-size: 13px !important;
        }

        section.main div[data-testid="stVerticalBlock"] > div:not(:has(.ng-nav-anchor)) button[kind="secondary"]:hover,
        div[data-testid="stDownloadButton"] button:hover {
            border-color: var(--bd-hover) !important;
            color: var(--tx-bright) !important;
        }

        /* =========================================================
           FULL-WIDTH BALANCED 4-COLUMN FOOTER (Zero dead space)
           ========================================================= */
        footer.ng-footer {
            display: block !important;
            visibility: visible !important;
            height: auto !important;
            position: relative !important;
            width: 100vw !important;
            max-width: 100vw !important;
            margin-left: calc(50% - 50vw) !important;
            margin-right: calc(50% - 50vw) !important;
            margin-top: 36px !important;
            margin-bottom: 0 !important;
            border-top: 1px solid var(--bd) !important;
            background: rgba(9, 9, 9, 0.88) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            color: var(--sub) !important;
        }

        .ng-footer-grid {
            max-width: 1140px;
            margin: 0 auto;
            padding: 26px max(20px, env(safe-area-inset-left, 20px)) 24px max(20px, env(safe-area-inset-right, 20px));
            display: grid;
            grid-template-columns: 1.55fr 1fr 1fr 1.05fr;
            gap: 28px;
            align-items: start;
        }

        .ng-footer-col {
            display: flex;
            flex-direction: column;
            gap: 7px;
            font-size: 13px;
            line-height: 1.45;
            min-width: 0;
        }

        .ng-footer-col .ng-mono {
            color: var(--tx-bright);
            margin-bottom: 3px;
        }

        .ng-footer-brand {
            display: inline-flex;
            align-items: center;
            gap: 9px;
            margin-bottom: 4px;
        }

        .ng-footer-bottom {
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .ng-footer-bottom-in {
            max-width: 1140px;
            margin: 0 auto;
            padding: 13px max(20px, env(safe-area-inset-left, 20px)) calc(13px + env(safe-area-inset-bottom, 0px)) max(20px, env(safe-area-inset-right, 20px));
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        /* =========================================================
           RESPONSIVE BREAKPOINTS (Tablet, Android, iOS)
           ========================================================= */
        @media (max-width: 1024px) {
            .ng-kpi-grid--5 {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .ng-kpi-grid--5 .ng-kpi:last-child {
                grid-column: 1 / -1;
            }
            .ng-kpi-grid--4 {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .ng-footer-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 22px 28px;
            }
        }

        @media (max-width: 860px) {
            div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) {
                flex-wrap: wrap !important;
                gap: 6px 8px !important;
                padding-top: calc(8px + env(safe-area-inset-top, 0px)) !important;
                padding-bottom: 8px !important;
                padding-left: max(14px, env(safe-area-inset-left, 14px)) !important;
                padding-right: max(14px, env(safe-area-inset-right, 14px)) !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="stColumn"]:first-child,
            div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="column"]:first-child {
                flex: 1 1 100% !important;
                padding-bottom: 4px !important;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="stColumn"]:not(:first-child),
            div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) > div[data-testid="column"]:not(:first-child) {
                flex: 1 1 calc(50% - 6px) !important;
                min-width: 0 !important;
                justify-content: flex-start !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) [data-testid="stButton"],
            div[data-testid="stHorizontalBlock"]:has(.ng-nav-anchor) button {
                width: 100% !important;
                justify-content: flex-start !important;
                font-size: 12.5px !important;
                min-height: 34px !important;
                height: 34px !important;
                padding: 4px 2px !important;
            }
            [data-testid="stMainBlockContainer"],
            .block-container {
                padding-left: max(14px, env(safe-area-inset-left, 14px)) !important;
                padding-right: max(14px, env(safe-area-inset-right, 14px)) !important;
            }
            div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) {
                flex-direction: column !important;
            }
            div[data-testid="stHorizontalBlock"]:not(:has(.ng-nav-anchor)) > div[data-testid="stColumn"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
            .ng-fi-row {
                grid-template-columns: minmax(90px, 130px) 1fr 48px;
                gap: 8px;
            }
            .ng-cm-grid {
                grid-template-columns: 64px repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 520px) {
            .ng-footer-grid {
                grid-template-columns: 1fr;
                gap: 18px;
            }
            .ng-brand-tag {
                display: none;
            }
            .ng-kpi {
                min-height: 86px;
                padding: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_dithered_shader_bg_and_layout_sync(session_hex: str) -> None:
    """Mount animated WebGL DitheredShaderBackground (with Canvas2D fallback) & navbar height sync."""
    components.html(
        f"""
        <script>
        (function() {{
            const parentWin = window.parent;
            const parentDoc = parentWin.document;

            // 1. Clean session URL mask with safe unload restore
            const targetPath = "/app/{session_hex}?hl=id";
            try {{
                if (parentWin.location.pathname !== "/app/{session_hex}") {{
                    parentWin.history.replaceState(null, "", targetPath);
                }}
                if (!parentWin.__ngUnloadBound) {{
                    parentWin.__ngUnloadBound = true;
                    parentWin.addEventListener("beforeunload", function() {{
                        try {{
                            parentWin.history.replaceState(null, "", "/");
                        }} catch (e) {{}}
                    }});
                }}
            }} catch (e) {{}}

            // 2. Keep --ng-nav-h synced to actual rendered navbar height across all viewports
            function syncNavHeight() {{
                const blocks = parentDoc.querySelectorAll('div[data-testid="stHorizontalBlock"]');
                for (let i = 0; i < blocks.length; i++) {{
                    if (blocks[i].querySelector(".ng-nav-anchor")) {{
                        const h = Math.ceil(blocks[i].getBoundingClientRect().height);
                        if (h > 0) {{
                            parentDoc.documentElement.style.setProperty("--ng-nav-h", h + "px");
                        }}
                        break;
                    }}
                }}
            }}
            syncNavHeight();
            if (!parentWin.__ngNavObserver) {{
                const ro = new ResizeObserver(syncNavHeight);
                ro.observe(parentDoc.body);
                parentWin.__ngNavObserver = ro;
                parentWin.addEventListener("resize", syncNavHeight);
            }}

            // 3. Mount persistent DitheredShaderBackground canvas once across Streamlit reruns
            if (parentWin.__ngDitherBgRunning && parentDoc.getElementById("ng-dither-bg-canvas")) {{
                return;
            }}
            parentWin.__ngDitherBgRunning = true;

            let canvas = parentDoc.getElementById("ng-dither-bg-canvas");
            if (!canvas) {{
                canvas = parentDoc.createElement("canvas");
                canvas.id = "ng-dither-bg-canvas";
                Object.assign(canvas.style, {{
                    position: "fixed",
                    inset: "0",
                    width: "100%",
                    height: "100dvh",
                    zIndex: "0",
                    pointerEvents: "none",
                    display: "block"
                }});
                parentDoc.body.prepend(canvas);
            }}

            const reducedMotion = parentWin.matchMedia &&
                parentWin.matchMedia("(prefers-reduced-motion: reduce)").matches;

            const gl = canvas.getContext("webgl", {{
                alpha: false,
                antialias: false,
                powerPreference: "low-power"
            }}) || canvas.getContext("experimental-webgl", {{
                alpha: false,
                antialias: false
            }});

            if (gl) {{
                const vertSrc = `
                    attribute vec2 a_pos;
                    void main() {{
                        gl_Position = vec4(a_pos, 0.0, 1.0);
                    }}
                `;
                const fragSrc = `
                    precision highp float;
                    uniform vec2 u_resolution;
                    uniform float u_time;
                    uniform float u_pixelSize;

                    float hash21(vec2 p) {{
                        p = fract(p * vec2(123.34, 456.21));
                        p += dot(p, p + 45.32);
                        return fract(p.x * p.y);
                    }}

                    float noise(vec2 p) {{
                        vec2 i = floor(p);
                        vec2 f = fract(p);
                        vec2 u = f * f * (3.0 - 2.0 * f);
                        float a = hash21(i);
                        float b = hash21(i + vec2(1.0, 0.0));
                        float c = hash21(i + vec2(0.0, 1.0));
                        float d = hash21(i + vec2(1.0, 1.0));
                        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
                    }}

                    float fbm(vec2 p) {{
                        float v = 0.0;
                        float a = 0.5;
                        mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
                        for (int i = 0; i < 4; i++) {{
                            v += a * noise(p);
                            p = rot * p * 2.02;
                            a *= 0.5;
                        }}
                        return v;
                    }}

                    float bayer4(vec2 cell) {{
                        int x = int(mod(cell.x, 4.0));
                        int y = int(mod(cell.y, 4.0));
                        int idx = y * 4 + x;
                        if (idx == 0) return 0.0 / 16.0;
                        if (idx == 1) return 8.0 / 16.0;
                        if (idx == 2) return 2.0 / 16.0;
                        if (idx == 3) return 10.0 / 16.0;
                        if (idx == 4) return 12.0 / 16.0;
                        if (idx == 5) return 4.0 / 16.0;
                        if (idx == 6) return 14.0 / 16.0;
                        if (idx == 7) return 6.0 / 16.0;
                        if (idx == 8) return 3.0 / 16.0;
                        if (idx == 9) return 11.0 / 16.0;
                        if (idx == 10) return 1.0 / 16.0;
                        if (idx == 11) return 9.0 / 16.0;
                        if (idx == 12) return 15.0 / 16.0;
                        if (idx == 13) return 7.0 / 16.0;
                        if (idx == 14) return 13.0 / 16.0;
                        return 5.0 / 16.0;
                    }}

                    void main() {{
                        vec2 cell = floor(gl_FragCoord.xy / u_pixelSize);
                        vec2 snapped = (cell + 0.5) * u_pixelSize;
                        vec2 uv = (snapped - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);

                        float t = u_time * 0.16;
                        vec2 q = vec2(
                            fbm(uv * 1.45 + vec2(t * 0.7, -t * 0.4)),
                            fbm(uv * 1.45 + vec2(-t * 0.5, t * 0.6) + vec2(3.2, 1.7))
                        );
                        float field = fbm(uv * 1.75 + 1.35 * q + vec2(0.0, t * 0.25));

                        // Flowing wave interference rings
                        float wave = 0.5 + 0.5 * sin(field * 8.5 - t * 2.2 + uv.x * 1.6 - uv.y * 1.2);
                        float lum = mix(field, wave, 0.42);

                        // Subtle horizon glow + gentle center vignette for effortless text contrast
                        float edge = smoothstep(0.18, 1.05, length(uv * vec2(0.85, 1.05)));
                        lum *= mix(0.48, 0.92, edge);
                        lum = clamp(lum * 0.62, 0.0, 1.0);

                        float threshold = bayer4(cell);
                        float dither = step(threshold, lum);

                        vec3 colBack = vec3(0.031, 0.031, 0.031);   // #080808
                        vec3 colFront = vec3(0.24, 0.24, 0.26);     // crisp subtle silver dither
                        vec3 col = mix(colBack, colFront, dither);
                        gl_FragColor = vec4(col, 1.0);
                    }}
                `;

                function compile(type, src) {{
                    const sh = gl.createShader(type);
                    gl.shaderSource(sh, src);
                    gl.compileShader(sh);
                    return sh;
                }}

                const prog = gl.createProgram();
                gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertSrc));
                gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragSrc));
                gl.linkProgram(prog);
                gl.useProgram(prog);

                const buf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.bufferData(
                    gl.ARRAY_BUFFER,
                    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
                    gl.STATIC_DRAW
                );
                const loc = gl.getAttribLocation(prog, "a_pos");
                gl.enableVertexAttribArray(loc);
                gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

                const uRes = gl.getUniformLocation(prog, "u_resolution");
                const uTime = gl.getUniformLocation(prog, "u_time");
                const uPix = gl.getUniformLocation(prog, "u_pixelSize");

                function resizeGL() {{
                    const dpr = Math.min(parentWin.devicePixelRatio || 1, 2);
                    const w = Math.max(1, Math.floor(parentWin.innerWidth * dpr));
                    const h = Math.max(1, Math.floor(parentWin.innerHeight * dpr));
                    if (canvas.width !== w || canvas.height !== h) {{
                        canvas.width = w;
                        canvas.height = h;
                    }}
                    gl.viewport(0, 0, canvas.width, canvas.height);
                    gl.uniform1f(uPix, 3.0 * dpr);
                }}

                resizeGL();
                parentWin.addEventListener("resize", resizeGL);

                const start = performance.now();
                function renderGL(now) {{
                    if (!parentDoc.hidden) {{
                        gl.uniform2f(uRes, canvas.width, canvas.height);
                        gl.uniform1f(uTime, (now - start) / 1000.0);
                        gl.drawArrays(gl.TRIANGLES, 0, 6);
                    }}
                    if (!reducedMotion) {{
                        parentWin.requestAnimationFrame(renderGL);
                    }}
                }}
                parentWin.requestAnimationFrame(renderGL);
            }} else {{
                // 2D Canvas Bayer 4x4 animated fallback for devices without WebGL
                const ctx = canvas.getContext("2d");
                const M = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
                canvas.style.imageRendering = "pixelated";

                function draw2D(ts) {{
                    const S = 4;
                    const w = (canvas.width = Math.max(1, Math.ceil(parentWin.innerWidth / S)));
                    const h = (canvas.height = Math.max(1, Math.ceil(parentWin.innerHeight / S)));
                    const im = ctx.createImageData(w, h);
                    const t = ts * 0.0005;
                    for (let j = 0; j < h; j++) {{
                        for (let i = 0; i < w; i++) {{
                            const nx = i / w - 0.5;
                            const ny = j / h - 0.5;
                            const v = 0.5 + 0.5 * Math.sin(nx * 6.0 + t + Math.cos(ny * 5.0 - t * 0.8));
                            const lum = v * 0.35;
                            const thr = M[(j % 4) * 4 + (i % 4)] / 16.0;
                            const on = lum > thr;
                            const k = (j * w + i) * 4;
                            const c = on ? 56 : 8;
                            im.data[k] = c;
                            im.data[k + 1] = c;
                            im.data[k + 2] = c + (on ? 4 : 0);
                            im.data[k + 3] = 255;
                        }}
                    }}
                    ctx.putImageData(im, 0, 0);
                    if (!reducedMotion) {{
                        parentWin.requestAnimationFrame(draw2D);
                    }}
                }}
                parentWin.requestAnimationFrame(draw2D);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


@st.cache_data
def load_json_report(rel_path: str) -> dict[str, Any] | list[Any] | None:
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        return None
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_predictor(model_variant: str) -> NetGuardPredictor:
    return NetGuardPredictor(
        models_dir=PROJECT_ROOT / "artifacts" / "models",
        model_variant=model_variant,
    )


def render_kpi_row(items: list[tuple[str, str, str]]) -> None:
    grid_cls = "ng-kpi-grid--4" if len(items) == 4 else "ng-kpi-grid--5"
    cards = "".join(
        f'<div class="ng-kpi">'
        f'<div class="ng-mono" style="color:var(--sub)">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(value)}</div>'
        f'<div class="s">{html.escape(sub)}</div>'
        f"</div>"
        for label, value, sub in items
    )
    st.markdown(f'<div class="ng-kpi-grid {grid_cls}">{cards}</div>', unsafe_allow_html=True)


def panel_html(num: int, title: str, meta: str, body_html: str) -> str:
    meta_html = f"<span>{html.escape(meta)}</span>" if meta else "<span>&nbsp;</span>"
    return (
        f'<div class="ng-panel">'
        f'<h3><div><i class="ng-num">{num:02d}</i>{html.escape(title)}</div>{meta_html}</h3>'
        f'<div class="ng-panel-body">{body_html}</div>'
        f"</div>"
    )


def table_html(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    tr = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="ng-sc"><table class="ng-tbl"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def bars_html(items: list[tuple[str, int, str]]) -> str:
    total = sum(v for _, v, _ in items) or 1
    out = []
    for label, count, color in items:
        pct = (count / total) * 100
        out.append(
            f'<div class="ng-bar">'
            f'<span class="ng-mono" style="color:var(--tx)">{html.escape(label)}</span>'
            f'<div class="ng-bar-track"><div class="ng-bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span class="ng-mono" style="text-align:right;color:var(--sub)">{count:,} ({pct:.1f}%)</span>'
            f"</div>"
        )
    return "".join(out)


def confusion_matrix_html(cm: dict[str, int]) -> str:
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = max(1, tn + fp + fn + tp)

    def _cell(tag: str, val: int, emphasize: bool) -> str:
        pct = (val / total) * 100
        if emphasize:
            style = "background:rgba(245,245,245,0.92);color:#080808;border-color:#f5f5f5;"
        else:
            style = "background:rgba(255,255,255,0.03);color:var(--tx-bright);"
        return (
            f'<div class="ng-cm-cell" style="{style}">'
            f'<span class="k">{tag}</span>'
            f'<span class="n">{val:,}</span>'
            f'<span class="p">{pct:.1f}%</span>'
            f"</div>"
        )

    return (
        '<div class="ng-cm-wrap">'
        '<div class="ng-cm-grid">'
        '<div></div>'
        '<div class="ng-cm-head">Pred · Normal</div>'
        '<div class="ng-cm-head">Pred · Anomaly</div>'
        '<div class="ng-cm-rowlabel">Actual<br>Normal</div>'
        f'{_cell("TN", tn, True)}'
        f'{_cell("FP", fp, False)}'
        '<div class="ng-cm-rowlabel">Actual<br>Anomaly</div>'
        f'{_cell("FN", fn, False)}'
        f'{_cell("TP", tp, True)}'
        "</div>"
        "</div>"
    )


def feature_importance_html(top_features: list[dict[str, Any]]) -> str:
    feats = top_features[:10]
    max_imp = max((float(f["importance"]) for f in feats), default=1.0) or 1.0
    rows = []
    for item in feats:
        name = html.escape(str(item["feature"]))
        imp = float(item["importance"])
        pct = (imp / max_imp) * 100
        rows.append(
            f'<div class="ng-fi-row">'
            f'<span class="ng-fi-name" title="{name}">{name}</span>'
            f'<div class="ng-fi-track"><div class="ng-fi-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="ng-fi-val">{imp:.4f}</span>'
            f"</div>"
        )
    return "".join(rows)


def render_navbar() -> None:
    cols = st.columns([3.0, 1.35, 1.25, 1.35, 1.3])
    with cols[0]:
        st.markdown(
            f"""
            <div class="ng-nav-anchor">
                <div class="ng-pixel-badge" title="NetGuard">{build_ng_pixel_logo_svg()}</div>
                <div class="ng-brand-text">
                    <span class="ng-brand-name">netguard.</span>
                    <span class="ng-brand-tag">NIDS</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    active = st.session_state["active_tab"]
    for idx, label in enumerate(NAV_ITEMS):
        with cols[idx + 1]:
            if st.button(
                label,
                key=f"nav_btn_{idx}",
                type="primary" if active == label else "secondary",
            ):
                st.session_state["active_tab"] = label
                st.rerun()


def render_footer(model_variant: str, models_ready: bool) -> None:
    status_pills = (
        '<span class="ng-pill ok">App running</span>'
        '<span class="ng-pill ok">Artifacts ready</span>'
        if models_ready
        else '<span class="ng-pill warn">Artifacts missing</span>'
    )
    logo_white = build_ng_pixel_logo_svg(fill="#f5f5f5", step=1.25, size=1.15)
    st.markdown(
        f"""
        <footer class="ng-footer">
            <div class="ng-footer-grid">
                <div class="ng-footer-col">
                    <div class="ng-footer-brand">
                        {logo_white}
                        <span class="ng-brand-name">netguard.</span>
                    </div>
                    <span>Leak-free XGBoost &amp; SMOTENC Network Intrusion Detection System evaluated on the NSL-KDD benchmark.</span>
                </div>
                <div class="ng-footer-col">
                    <span class="ng-mono">Views</span>
                    <span>Overview Dashboard</span>
                    <span>Dataset Analysis</span>
                    <span>Model Performance</span>
                    <span>Intrusion Scanner</span>
                </div>
                <div class="ng-footer-col">
                    <span class="ng-mono">Operational Scope</span>
                    <span>Offline CSV inference only</span>
                    <span>No live packet capture</span>
                    <span>Max 20 records per scan</span>
                    <span>Binary: Normal / Anomaly</span>
                </div>
                <div class="ng-footer-col">
                    <span class="ng-mono">Runtime Status</span>
                    <div>{status_pills}</div>
                    <span>Active model: <code style="color:var(--tx-bright)">{html.escape(model_variant)}</code></span>
                    <span>Benchmark: <code style="color:var(--tx-bright)">KDDTest+</code></span>
                </div>
            </div>
            <div class="ng-footer-bottom">
                <div class="ng-footer-bottom-in ng-mono">
                    <span>© 2026 NetGuard · Offline NIDS</span>
                    <a href="#" style="color:var(--tx);text-decoration:none" onclick="window.scrollTo({{top:0,behavior:'smooth'}});return false;">Back to top ↑</a>
                </div>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    if "session_hex" not in st.session_state:
        st.session_state["session_hex"] = secrets.token_hex(8)
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Overview Dashboard"
    if "model_variant" not in st.session_state:
        st.session_state["model_variant"] = "xgb_smotenc"
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.50
    if "scan_history" not in st.session_state:
        st.session_state["scan_history"] = []
    if "last_scan" not in st.session_state:
        st.session_state["last_scan"] = None

    inject_theme_and_styles()
    inject_dithered_shader_bg_and_layout_sync(st.session_state["session_hex"])

    results = load_json_report("reports/metrics/experiment_results.json")
    eda_summary = load_json_report("reports/metrics/eda_summary.json")
    top_corr = load_json_report("reports/metrics/top_correlations.json")

    models_dir = PROJECT_ROOT / "artifacts" / "models"
    models_ready = (models_dir / "preprocessor.joblib").exists() and (
        models_dir / "xgb_smotenc.joblib"
    ).exists()

    render_navbar()

    active_tab = st.session_state["active_tab"]
    model_variant = st.session_state["model_variant"]
    threshold = st.session_state["threshold"]

    # Hero Section
    st.markdown(
        f"""
        <div class="ng-hero">
            <span class="ng-tag ng-mono"><span class="ng-dot"></span>netguard // offline nids</span>
            <div class="ng-h1">{html.escape(active_tab.upper())}</div>
            <p class="ng-sub">{html.escape(NAV_SUBTITLES[active_tab])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 01. OVERVIEW DASHBOARD
    # -------------------------------------------------------------------------
    if active_tab == "Overview Dashboard":
        history = st.session_state["scan_history"]
        total_scanned = sum(item["flows"] for item in history)
        total_normal = sum(item["normal"] for item in history)
        total_attack = sum(item["attacks"] for item in history)

        render_kpi_row(
            [
                ("Total analyzed", f"{total_scanned:,}", "records scanned in session"),
                ("Normal", f"{total_normal:,}", "predicted benign flows"),
                ("Anomaly", f"{total_attack:,}", "predicted intrusion flows"),
                (
                    "Model",
                    "SMOTENC" if model_variant == "xgb_smotenc" else "Baseline",
                    model_variant,
                ),
                ("Pipeline", "Ready" if models_ready else "Missing", "Streamlit inference"),
            ]
        )

        col1, col2 = st.columns(2)
        with col1:
            if total_scanned > 0:
                dist_body = bars_html(
                    [
                        ("normal", total_normal, "#e5e5e5"),
                        ("anomaly", total_attack, "#ff6161"),
                    ]
                )
            else:
                dist_body = '<div class="ng-empty"><b>[ ]</b>No scans yet. Run a batch in Intrusion Scanner.</div>'
            st.markdown(
                panel_html(1, "Detection distribution", "session predictions", dist_body),
                unsafe_allow_html=True,
            )

        with col2:
            status_rows = [
                ["Runtime", '<span class="ng-pill ok">ONLINE</span>'],
                ["Active model", f"<code>{html.escape(model_variant)}.joblib</code>"],
                ["Benchmark", "NSL-KDD (KDDTrain+ / KDDTest+)"],
                ["Input mode", "Offline CSV upload (max 20 rows)"],
            ]
            st.markdown(
                panel_html(
                    2,
                    "System status",
                    "deployment",
                    table_html(["item", "value"], status_rows),
                ),
                unsafe_allow_html=True,
            )

        if history:
            hist_rows = [
                [
                    html.escape(h["timestamp"]),
                    html.escape(h["source"]),
                    html.escape(h["model"]),
                    f"{h['threshold']:.2f}",
                    str(h["flows"]),
                    f"<span style='color:var(--ok)'>{h['normal']}</span>",
                    f"<span style='color:var(--crit)'>{h['attacks']}</span>",
                ]
                for h in reversed(history)
            ]
            hist_body = table_html(
                ["Time", "Source", "Model", "Cutoff", "Flows", "Normal", "Anomaly"],
                hist_rows,
            )
        else:
            hist_body = '<div class="ng-empty"><b>[ ]</b>No scan activity recorded in this session.</div>'

        st.markdown(
            panel_html(3, "Scan history", "current session", hist_body),
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # 02. DATASET ANALYSIS
    # -------------------------------------------------------------------------
    elif active_tab == "Dataset Analysis":
        train_rows = eda_summary["train_rows"] if eda_summary else 125973
        test_rows = eda_summary["test_rows"] if eda_summary else 22544
        feat_count = eda_summary["feature_count"] if eda_summary else 41

        render_kpi_row(
            [
                ("Dataset", "NSL-KDD", "KDDTrain+ & KDDTest+"),
                ("Total records", f"{train_rows + test_rows:,}", f"{train_rows:,} train / {test_rows:,} test"),
                ("Raw features", str(feat_count), "38 numeric + 3 categorical"),
                ("Missing values", "0", "verified across splits"),
                ("Duplicates", "0", "deduplicated benchmark"),
            ]
        )

        col1, col2 = st.columns(2)
        with col1:
            if eda_summary:
                tr_norm = eda_summary["train_binary_distribution"]["0"]
                tr_att = eda_summary["train_binary_distribution"]["1"]
                te_norm = eda_summary["test_binary_distribution"]["0"]
                te_att = eda_summary["test_binary_distribution"]["1"]
                body_dist = (
                    '<div class="ng-mono" style="color:var(--sub);margin-bottom:4px">KDDTrain+ (125,973)</div>'
                    + bars_html([("normal", tr_norm, "#e5e5e5"), ("anomaly", tr_att, "#ff6161")])
                    + '<div class="ng-mono" style="color:var(--sub);margin:14px 0 4px">KDDTest+ (22,544)</div>'
                    + bars_html([("normal", te_norm, "#e5e5e5"), ("anomaly", te_att, "#ff6161")])
                )
            else:
                body_dist = '<div class="ng-empty"><b>[ ]</b>eda_summary.json not found.</div>'
            st.markdown(
                panel_html(1, "Class distribution", "ground truth", body_dist),
                unsafe_allow_html=True,
            )

        with col2:
            if top_corr:
                corr_rows = [
                    [
                        html.escape(str(c["feature_1"])),
                        html.escape(str(c["feature_2"])),
                        f"<span style='color:var(--tx-bright)'>{c['absolute_correlation']:.4f}</span>",
                    ]
                    for c in top_corr[:8]
                ]
                corr_body = table_html(["feature_1", "feature_2", "|r|"], corr_rows)
            else:
                corr_body = '<div class="ng-empty"><b>[ ]</b>top_correlations.json not found.</div>'
            st.markdown(
                panel_html(2, "Feature correlations", "|r| > 0.90", corr_body),
                unsafe_allow_html=True,
            )

        col3, col4 = st.columns(2)
        with col3:
            sample_path = PROJECT_ROOT / "data" / "processed" / "sample_scanner_20rows.csv"
            if sample_path.exists():
                sdf = pd.read_csv(sample_path).head(8)
                srows = [
                    [
                        str(i + 1),
                        html.escape(str(r["protocol_type"])),
                        html.escape(str(r["service"])),
                        html.escape(str(r["flag"])),
                        f"{int(r['src_bytes']):,}",
                        f"{int(r['dst_bytes']):,}",
                        html.escape(str(r["label"])),
                    ]
                    for i, r in sdf.iterrows()
                ]
                sample_body = table_html(
                    ["#", "protocol", "service", "flag", "src_bytes", "dst_bytes", "label"],
                    srows,
                )
            else:
                sample_body = '<div class="ng-empty"><b>[ ]</b>Sample CSV not generated yet.</div>'
            st.markdown(
                panel_html(3, "Sample records", "KDDTest+ slice", sample_body),
                unsafe_allow_html=True,
            )

        with col4:
            smote_rows = [
                ["Train Split (80%)", "53,874", "46,904", "100,778"],
                ["After SMOTENC (Train)", "53,874", "53,874", "107,748"],
                ["Validation Split (20%)", "13,469", "11,726", "25,195"],
                ["External KDDTest+", "9,711", "12,833", "22,544"],
            ]
            st.markdown(
                panel_html(
                    4,
                    "SMOTENC resampling",
                    "train split only",
                    table_html(["partition", "normal", "anomaly", "total"], smote_rows),
                ),
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # 03. MODEL PERFORMANCE
    # -------------------------------------------------------------------------
    elif active_tab == "Model Performance":
        if results is None:
            st.markdown(
                panel_html(
                    1,
                    "Evaluation artifacts",
                    "missing",
                    '<div class="ng-empty"><b>[ ]</b>Run <code>PYTHONPATH=src python -m scripts.train</code> first.</div>',
                ),
                unsafe_allow_html=True,
            )
        else:
            exp_a = results["experiments"]["xgb_baseline"]["external_test"]
            exp_b = results["experiments"]["xgb_smotenc"]["external_test"]

            render_kpi_row(
                [
                    ("Algorithm", "XGBoost", "121 OHE features"),
                    ("Split protocol", "Leak-free", "KDDTrain+ 80/20"),
                    (
                        "Accuracy",
                        f"{exp_b['accuracy']*100:.2f}%",
                        f"{(exp_b['accuracy']-exp_a['accuracy'])*100:+.2f}% vs baseline",
                    ),
                    (
                        "Precision / Recall",
                        f"{exp_b['precision']*100:.1f}% / {exp_b['recall']*100:.1f}%",
                        "KDDTest+ external",
                    ),
                    (
                        "F1-Score",
                        f"{exp_b['f1_score']:.4f}",
                        f"ROC-AUC {exp_b['roc_auc']:.4f}",
                    ),
                ]
            )

            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                st.markdown(
                    panel_html(
                        1,
                        "Confusion matrix · Baseline",
                        "Exp A · KDDTest+",
                        confusion_matrix_html(exp_a["confusion_matrix"]),
                    ),
                    unsafe_allow_html=True,
                )
            with cm_col2:
                st.markdown(
                    panel_html(
                        2,
                        "Confusion matrix · SMOTENC",
                        "Exp B · KDDTest+",
                        confusion_matrix_html(exp_b["confusion_matrix"]),
                    ),
                    unsafe_allow_html=True,
                )

            comp_rows = []
            for label, key in [
                ("Accuracy", "accuracy"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1-Score", "f1_score"),
                ("ROC-AUC", "roc_auc"),
                ("PR-AUC", "pr_auc"),
                ("False Positive Rate", "fpr"),
                ("False Negative Rate", "fnr"),
            ]:
                va = exp_a[key]
                vb = exp_b[key]
                diff = vb - va
                improved = (diff >= 0 and key not in ("fpr", "fnr")) or (
                    diff <= 0 and key in ("fpr", "fnr")
                )
                diff_color = "var(--ok)" if improved else "var(--warn)"
                comp_rows.append(
                    [
                        label,
                        f"{va:.4f}",
                        f"<strong>{vb:.4f}</strong>",
                        f"<span style='color:{diff_color}'>{diff:+.4f}</span>",
                    ]
                )

            st.markdown(
                panel_html(
                    3,
                    "Baseline vs SMOTENC",
                    "KDDTest+ · 22,544 records",
                    table_html(
                        ["metric", "baseline", "smotenc", "delta"],
                        comp_rows,
                    ),
                ),
                unsafe_allow_html=True,
            )

            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown(
                    panel_html(
                        4,
                        "Feature importance",
                        "Top 10 · Gain",
                        feature_importance_html(
                            results["experiments"]["xgb_smotenc"]["top_20_features"]
                        ),
                    ),
                    unsafe_allow_html=True,
                )

            with f_col2:
                cat_a = exp_a.get("category_breakdown", {})
                cat_b = exp_b.get("category_breakdown", {})
                cat_rows = [
                    [
                        html.escape(cat),
                        f"{cat_b[cat]['total']:,}",
                        f"{cat_a.get(cat, {}).get('correct_rate', 0)*100:.2f}%",
                        f"<strong style='color:var(--tx-bright)'>{cat_b[cat]['correct_rate']*100:.2f}%</strong>",
                    ]
                    for cat in sorted(cat_b.keys())
                ]
                st.markdown(
                    panel_html(
                        5,
                        "Detection rate by category",
                        "specificity / recall",
                        table_html(
                            ["category", "samples", "baseline (%)", "smotenc (%)"],
                            cat_rows,
                        ),
                    ),
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # 04. INTRUSION SCANNER
    # -------------------------------------------------------------------------
    elif active_tab == "Intrusion Scanner":
        st.markdown(
            """
            <div class="ng-ok-card">
                <span class="ng-mono" style="color:var(--warn)">Batch inference ready</span>
                <h4>Scan NSL-KDD CSV records offline</h4>
                <span style="color:var(--sub)">
                    Accepts 41-feature unlabeled CSVs or 42/43-column labeled NSL-KDD slices (max 20 rows per scan). Predictions are kept separate from ground-truth labels.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ctrl1, ctrl2 = st.columns([1.4, 1.6])
        with ctrl1:
            chosen_model = st.selectbox(
                "Model Variant",
                options=["xgb_smotenc", "xgb_baseline"],
                index=0 if model_variant == "xgb_smotenc" else 1,
                format_func=lambda x: (
                    "XGBoost + SMOTENC (Recommended)"
                    if x == "xgb_smotenc"
                    else "XGBoost Baseline"
                ),
            )
            st.session_state["model_variant"] = chosen_model

        with ctrl2:
            chosen_thresh = st.slider(
                "Decision Alert Threshold",
                min_value=0.10,
                max_value=0.90,
                value=float(threshold),
                step=0.05,
            )
            st.session_state["threshold"] = float(chosen_thresh)

        up_col, opt_col = st.columns([2.1, 1.4])
        with up_col:
            uploaded = st.file_uploader(
                "Upload CSV file (max 20 rows)",
                type=["csv", "txt"],
                label_visibility="collapsed",
            )
        with opt_col:
            drop_na = st.checkbox("Drop records with missing values", value=True)
            b1, b2 = st.columns(2)
            with b1:
                run_clicked = st.button("Run scan", type="primary", use_container_width=True)
            with b2:
                sample_clicked = st.button(
                    "Load sample", type="secondary", use_container_width=True
                )

        if uploaded is not None:
            st.session_state["scanner_input_df"] = pd.read_csv(uploaded)
            st.session_state["scanner_input_name"] = uploaded.name
        elif sample_clicked or "scanner_input_df" not in st.session_state:
            default_sample = PROJECT_ROOT / "data" / "processed" / "sample_scanner_20rows.csv"
            if default_sample.exists():
                st.session_state["scanner_input_df"] = pd.read_csv(default_sample)
                st.session_state["scanner_input_name"] = "sample_scanner_20rows.csv"

        if (
            (run_clicked or sample_clicked or st.session_state["last_scan"] is None)
            and "scanner_input_df" in st.session_state
        ):
            try:
                batch_df = st.session_state["scanner_input_df"].copy()
                if drop_na:
                    batch_df = batch_df.dropna()

                predictor = get_predictor(st.session_state["model_variant"])
                summary = predictor.predict_dataframe(
                    batch_df,
                    threshold=st.session_state["threshold"],
                    max_rows=20,
                )
                st.session_state["last_scan"] = {
                    "summary": summary,
                    "source": st.session_state["scanner_input_name"],
                    "model": st.session_state["model_variant"],
                    "threshold": st.session_state["threshold"],
                }
                if run_clicked or sample_clicked:
                    st.session_state["scan_history"].append(
                        {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "source": st.session_state["scanner_input_name"],
                            "model": st.session_state["model_variant"],
                            "threshold": st.session_state["threshold"],
                            "flows": summary.total_flows,
                            "normal": summary.predicted_normal,
                            "attacks": summary.predicted_attacks,
                        }
                    )
            except Exception as exc:
                st.error(f"Scanner error: {exc}")

        last_scan = st.session_state["last_scan"]
        if last_scan is not None:
            summary = last_scan["summary"]
            res_df: pd.DataFrame = summary.results_df

            render_kpi_row(
                [
                    ("Scanned", str(summary.total_flows), last_scan["source"]),
                    (
                        "Anomaly",
                        str(summary.predicted_attacks),
                        f"{summary.attack_ratio*100:.1f}% of batch",
                    ),
                    ("Normal", str(summary.predicted_normal), "predicted benign"),
                    (
                        "Mean score",
                        f"{summary.avg_attack_probability:.4f}",
                        f"cutoff {last_scan['threshold']:.2f}",
                    ),
                ]
            )

            headers = [
                "#",
                "predicted",
                "score",
                "risk",
                "protocol",
                "service",
                "flag",
                "src_bytes",
                "dst_bytes",
                "count",
            ]
            has_gt = "ground_truth_label" in res_df.columns
            if has_gt:
                headers.extend(["ground_truth", "family", "status"])

            rows_formatted: list[list[str]] = []
            for _, r in res_df.iterrows():
                pred_pill = (
                    '<span class="ng-pill crit">anomaly</span>'
                    if r["prediction"] == "ATTACK"
                    else '<span class="ng-pill ok">normal</span>'
                )
                risk = str(r["risk_level"])
                risk_cls = (
                    "crit"
                    if risk in ("CRITICAL", "HIGH")
                    else ("warn" if risk == "MEDIUM" else "ok")
                )
                risk_pill = f'<span class="ng-pill {risk_cls}">{html.escape(risk)}</span>'

                row_cells = [
                    str(r["flow_id"]),
                    pred_pill,
                    f"{r['attack_probability']:.4f}",
                    risk_pill,
                    html.escape(str(r["protocol_type"])),
                    html.escape(str(r["service"])),
                    html.escape(str(r["flag"])),
                    f"{int(r['src_bytes']):,}",
                    f"{int(r['dst_bytes']):,}",
                    str(int(r["count"])),
                ]
                if has_gt:
                    chk_color = "var(--ok)" if r["is_correct"] == "MATCH" else "var(--warn)"
                    row_cells.extend(
                        [
                            html.escape(str(r["ground_truth_label"])),
                            html.escape(str(r["ground_truth_family"])),
                            f"<span style='color:{chk_color}'>{html.escape(str(r['is_correct']))}</span>",
                        ]
                    )
                rows_formatted.append(row_cells)

            st.markdown(
                panel_html(
                    1,
                    "Scan results",
                    "predicted class + probability score",
                    table_html(headers, rows_formatted),
                ),
                unsafe_allow_html=True,
            )

            st.download_button(
                label="Download Results CSV",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name=f"netguard_scan_{last_scan['model']}_t{int(last_scan['threshold']*100)}.csv",
                mime="text/csv",
            )
        else:
            st.markdown(
                panel_html(
                    1,
                    "Scan results",
                    "predicted class + probability score",
                    '<div class="ng-empty"><b>[ ]</b>Results and CSV download appear here after a scan.</div>',
                ),
                unsafe_allow_html=True,
            )

    render_footer(
        model_variant=model_variant,
        models_ready=models_ready,
    )


if __name__ == "__main__":
    main()
