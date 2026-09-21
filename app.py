"""
INNOVATRIX  |  Mine subsidence monitoring console
Prototype build, simulated sensor data.

Run:   streamlit run app.py
Needs: streamlit >= 1.40, plotly, pandas, numpy

Optional: drop a real site photo at  assets/site_banner.jpg  and it will be
used as the page-header image (a dark overlay is added so text stays readable).
"""

import base64
import math
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="INNOVATRIX | Mine monitoring console",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN TOKENS
# ============================================================

FONT = "IBM Plex Sans, Segoe UI, Arial, sans-serif"
INK = "#14202B"
MUTED = "#5F6E7B"
LINE = "#D9DFE5"
GRID = "#E8ECF0"
STEEL = "#2B6C8F"

STATUS = {
    "Normal": {"color": "#2F7D5B", "soft": "#E4F1EA"},
    "Watch": {"color": "#B7791F", "soft": "#FBF0D9"},
    "Warning": {"color": "#B4372E", "soft": "#F9E4E1"},
}

# Prototype thresholds. These must be calibrated with real mine data.
WATCH_AT, WARNING_AT = 35, 60
TILT_BASE, DIST_REF = 1.8, 12.0
LIMITS = {
    "tilt": (2.4, 2.8),          # degrees
    "vibration": (11.0, 15.0),   # relative index, 0-20
    "distance": (13.0, 13.6),    # mm reading (reference is 12.0 mm)
}

WINDOWS = {"Last hour": 60, "Last 3 hours": 180, "Last 6 hours": 360}
PAGES = ["Overview", "Sensor trends", "Sensor nodes", "Site map", "Alerts and events"]

PAGE_COPY = {
    "Overview": (
        "Site overview",
        "Ground tilt, vibration and relative displacement across all monitoring nodes, "
        "with a combined risk index for early warning.",
    ),
    "Sensor trends": (
        "Sensor trends",
        "Compare readings between nodes over time and check them against the watch and warning levels.",
    ),
    "Sensor nodes": (
        "Sensor nodes",
        "Health, power and radio link status for each node reporting through the LoRa gateway.",
    ),
    "Site map": (
        "Site map",
        "Node and gateway positions above the extracted panel, with link distances.",
    ),
    "Alerts and events": (
        "Alerts and events",
        "Threshold crossings and system messages, with the scoring rules used to raise them.",
    ),
}

# Simulated nodes. "tilt", "vib" and "dist" are the values each node is
# simulated to be reading right now; the history is built backwards from them.
NODES = {
    "N-001": dict(zone="Zone A", where="Main haulage, north", color="#2B6C8F",
                  tilt=2.14, vib=7.6, dist=12.40, drift=(0.10, 0.00, 0.10),
                  batt=91, rssi=-71, seen=4, sx=250, mx=250, my=165,
                  fw="1.3.2", installed="12 Aug 2026", calibrated="02 Sep 2026"),
    "N-002": dict(zone="Zone B", where="Panel centre", color="#4F8F7B",
                  tilt=1.83, vib=7.0, dist=11.80, drift=(0.03, 0.00, -0.05),
                  batt=86, rssi=-84, seen=11, sx=520, mx=430, my=290,
                  fw="1.3.2", installed="12 Aug 2026", calibrated="02 Sep 2026"),
    "N-003": dict(zone="Zone C", where="Panel edge, east", color="#C2762B",
                  tilt=2.47, vib=9.5, dist=13.10, drift=(0.30, 1.6, 0.35),
                  batt=94, rssi=-92, seen=7, sx=690, mx=610, my=195,
                  fw="1.3.1", installed="14 Aug 2026", calibrated="03 Sep 2026"),
    "N-004": dict(zone="Zone D", where="Ventilation shaft", color="#7A6C9C",
                  tilt=1.92, vib=6.0, dist=12.00, drift=(0.05, 0.00, 0.03),
                  batt=79, rssi=-88, seen=22, sx=820, mx=790, my=335,
                  fw="1.3.2", installed="15 Aug 2026", calibrated="03 Sep 2026"),
}
GATEWAY_MAP = (70, 250)


# ============================================================
# SMALL HELPERS
# ============================================================

def H(s: str) -> str:
    """Collapse HTML to one line. Streamlit's markdown treats indented lines
    and blank lines as code blocks, which breaks raw HTML."""
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())


def md(s: str):
    st.markdown(H(s), unsafe_allow_html=True)


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def state_of(score: float) -> str:
    if score < WATCH_AT:
        return "Normal"
    if score < WARNING_AT:
        return "Watch"
    return "Warning"


def pill(s: str) -> str:
    return f'<span class="pill p-{s.lower()}"><i></i>{s}</span>'


def meter(value, watch, warn, top):
    pct = lambda v: max(0.0, min(100.0, v / top * 100))
    c = STATUS[
        "Normal" if value < watch else "Watch" if value < warn else "Warning"
    ]["color"]
    return (
        f'<div class="meter"><div class="meter-fill" style="width:{pct(value):.1f}%;background:{c}"></div>'
        f'<i style="left:{pct(watch):.1f}%"></i><i style="left:{pct(warn):.1f}%"></i></div>'
    )


_panel_count = 0


@contextmanager
def panel(title: str, meta: str = ""):
    """White panel with a title row. Styled through the st-key-panel_* class."""
    global _panel_count
    _panel_count += 1
    with st.container(key=f"panel_{_panel_count}"):
        md(f'<div class="ph"><span class="ph-t">{title}</span><span class="ph-m">{meta}</span></div>')
        yield


def show(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


# ============================================================
# CSS
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp, .stApp p, .stApp label, .stApp div, .stApp button, .stApp input,
.stApp td, .stApp th, .stApp li, .stMarkdown {
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
}
.stApp { background: #EDF0F3; color: #14202B; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, div[data-testid="stDecoration"], .stDeployButton { display: none; }
.block-container { max-width: 1480px; padding: 1.2rem 2rem 2.5rem 2rem; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: #12212B; border-right: 1px solid #223846; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #C9D6DD; }
.brand { display: flex; align-items: center; gap: 11px; padding: 2px 2px 18px 2px;
         border-bottom: 1px solid #26404F; margin-bottom: 14px; }
.brand-name { font-size: 18px; font-weight: 600; letter-spacing: .6px; color: #FFFFFF; line-height: 1.1; }
.brand-sub { font-size: 12px; color: #8FA6B2; margin-top: 3px; }
.side-h { font-size: 12px; font-weight: 500; color: #7F98A5; margin: 22px 0 8px 0; }
.side-kv { display: flex; justify-content: space-between; font-size: 13px; padding: 8px 0;
           border-bottom: 1px solid #1E3441; }
.side-kv span:first-child { color: #8FA6B2; }
.side-kv span:last-child { color: #E6EEF2; font-variant-numeric: tabular-nums; }
.side-state { display: flex; align-items: center; gap: 9px; font-size: 13px; color: #E6EEF2; padding: 4px 0; }
.side-state small { display: block; color: #8FA6B2; font-size: 12px; }

section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 9px 12px; border-radius: 5px; border-left: 3px solid transparent;
    cursor: pointer; width: 100%; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #1A3140; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #1E3847; border-left-color: #F2B705; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size: 14px; color: #C9D6DD; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #FFFFFF; font-weight: 600; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #1B3140; border: 1px solid #2E4A5A; }
section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #E6EEF2; }
section[data-testid="stSidebar"] div[data-baseweb="select"] svg { fill: #9FB4BF; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid #34505F; color: #E6EEF2; border-radius: 5px; }
section[data-testid="stSidebar"] .stButton > button:hover { border-color: #F2B705; color: #FFFFFF; }

/* ---------- page header ---------- */
.pagehead { border-radius: 6px; padding: 26px 30px; color: #FFFFFF; background-color: #12212B;
            background-size: cover; background-position: right center;
            display: flex; justify-content: space-between; align-items: flex-end;
            gap: 24px; margin-bottom: 18px; }
.crumb { font-size: 13px; color: #9FB6C2; margin-bottom: 7px; }
.pagetitle { font-size: 30px; font-weight: 600; letter-spacing: -.3px; line-height: 1.15; color: #FFFFFF; }
.pagehead p { font-size: 14px; color: #C5D4DB; margin: 9px 0 0 0; max-width: 620px; line-height: 1.55; }
.headstat { text-align: right; min-width: 250px; }
.hs-state { display: inline-flex; align-items: center; gap: 9px; font-size: 21px; font-weight: 600; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.hs-line { font-size: 13px; color: #C5D4DB; margin-top: 5px; font-variant-numeric: tabular-nums; }

/* ---------- KPI strip ---------- */
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); background: #FFFFFF;
        border: 1px solid #D9DFE5; border-radius: 6px; margin-bottom: 18px; }
.kpi { padding: 16px 20px 18px 20px; border-right: 1px solid #E3E8ED; }
.kpi:last-child { border-right: 0; }
.kpi-l { font-size: 13px; color: #5F6E7B; }
.kpi-v { font-size: 28px; font-weight: 600; letter-spacing: -.5px; margin-top: 6px;
         font-variant-numeric: tabular-nums; color: #14202B; }
.kpi-v small { font-size: 14px; font-weight: 400; color: #6B7A87; margin-left: 5px; letter-spacing: 0; }
.kpi-n { font-size: 12px; color: #6B7A87; margin-top: 9px; }
.meter { position: relative; height: 6px; background: #E6EBEF; border-radius: 3px; margin-top: 11px; }
.meter-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
.meter i { position: absolute; top: -2px; bottom: -2px; width: 1px; background: #8896A2; }

/* ---------- panels ---------- */
div[class*="st-key-panel_"] { background: #FFFFFF; border: 1px solid #D9DFE5; border-radius: 6px;
                              padding: 16px 18px 14px 18px; gap: .55rem; }
.ph { display: flex; justify-content: space-between; align-items: baseline;
      border-bottom: 1px solid #E6EAEE; padding-bottom: 10px; margin-bottom: 4px; }
.ph-t { font-size: 15px; font-weight: 600; color: #14202B; }
.ph-m { font-size: 12px; color: #7A8894; }
img.art { width: 100%; display: block; border-radius: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; font-size: 12px; color: #5F6E7B; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.legend .lg-note { margin-left: auto; color: #7A8894; }

/* ---------- node rows ---------- */
.nr { padding: 12px 0 13px 0; border-bottom: 1px solid #EEF1F4; }
.nr:last-child { border-bottom: 0; }
.nr-top { display: flex; justify-content: space-between; align-items: center; }
.nr-id { font-weight: 600; font-size: 14px; }
.nr-z { color: #6B7A87; font-size: 12px; margin-left: 8px; }
.nr-foot { display: flex; justify-content: space-between; font-size: 12px; color: #6B7A87; margin-top: 9px; }
.nr-foot b { color: #14202B; font-size: 13px; font-variant-numeric: tabular-nums; }

.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500;
        padding: 3px 10px 3px 8px; border-radius: 999px; }
.pill i { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.p-normal { background: #E4F1EA; color: #1F6B4A; } .p-normal i { background: #2F7D5B; }
.p-watch { background: #FBF0D9; color: #8A5A0F; }   .p-watch i { background: #B7791F; }
.p-warning { background: #F9E4E1; color: #8F2A22; } .p-warning i { background: #B4372E; }
.p-info { background: #E8EEF3; color: #3B5568; }    .p-info i { background: #6C8798; }

/* ---------- tables ---------- */
table.dt { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.dt th { text-align: left; font-weight: 500; color: #6B7A87; font-size: 12px; padding: 9px 10px;
              border-bottom: 1px solid #D9DFE5; background: #F6F8FA; }
table.dt td { padding: 12px 10px; border-bottom: 1px solid #EEF1F4; vertical-align: middle; color: #14202B; }
table.dt tr:last-child td { border-bottom: 0; }
table.dt .num { text-align: right; font-variant-numeric: tabular-nums; }
.sub { display: block; font-size: 12px; color: #7A8894; margin-top: 2px; }
.bat { display: inline-block; width: 64px; height: 8px; background: #E6EBEF; border-radius: 2px;
       vertical-align: middle; margin-right: 8px; position: relative; }
.bat b { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 2px; }
.sig { display: inline-flex; align-items: flex-end; gap: 2px; height: 14px; margin-right: 8px; vertical-align: middle; }
.sig i { width: 4px; background: #CBD3DA; display: block; }
.sig i:nth-child(1) { height: 5px; } .sig i:nth-child(2) { height: 8px; }
.sig i:nth-child(3) { height: 11px; } .sig i:nth-child(4) { height: 14px; }
.sig i.on { background: #2B6C8F; }

/* ---------- events ---------- */
.ev { display: grid; grid-template-columns: 92px 70px 1fr auto; gap: 14px; align-items: center;
      padding: 12px 0; border-bottom: 1px solid #EEF1F4; font-size: 13.5px; }
.ev:last-child { border-bottom: 0; }
.ev-t { font-variant-numeric: tabular-nums; color: #2A3944; }
.ev-t small { display: block; color: #8794A0; font-size: 12px; }
.ev-n { font-weight: 600; }
.ev-m { color: #2A3944; line-height: 1.45; }

/* ---------- key/value + notice ---------- */
.kv { display: grid; grid-template-columns: 150px 1fr; row-gap: 11px; font-size: 13.5px; }
.kv div:nth-child(odd) { color: #6B7A87; }
.kv div:nth-child(even) { color: #14202B; font-variant-numeric: tabular-nums; }
.notice { border-left: 3px solid #2B6C8F; background: #F1F6F9; padding: 12px 16px; font-size: 13px;
          color: #3B4A56; border-radius: 0 4px 4px 0; line-height: 1.6; margin-top: 4px; }
.foot { text-align: center; color: #8794A0; font-size: 12px; padding: 26px 0 2px 0; }

/* ---------- widgets ---------- */
span[data-baseweb="tag"] { background: #2B6C8F; }
span[data-baseweb="tag"] span { color: #FFFFFF; }
.stDownloadButton > button, .stButton > button { border-radius: 5px; }
div[data-testid="stDataFrame"] { border: 1px solid #D9DFE5; border-radius: 4px; }

@media (max-width: 1100px) {
    .kpis { grid-template-columns: repeat(2, 1fr); }
    .pagehead { flex-direction: column; align-items: flex-start; }
    .headstat { text-align: left; }
}
"""

st.markdown(f"<style>{H(CSS)}</style>", unsafe_allow_html=True)

# ============================================================
# LOGIN THEME
# ============================================================
LOGIN_CSS = """
.login-brand{min-height:500px;padding:48px 42px;border-radius:8px 0 0 8px;background:#12212B;color:#FFFFFF;position:relative;overflow:hidden}
.login-brand:before,.login-brand:after{content:"";position:absolute;border:1px solid rgba(157,194,211,.18);border-radius:50%;pointer-events:none}
.login-brand:before{width:600px;height:600px;right:-330px;top:-160px}.login-brand:after{width:390px;height:390px;right:-170px;bottom:-190px}
.login-brand-content{position:relative;z-index:2}.login-logo-row{display:flex;align-items:center;gap:12px;margin-bottom:54px}
.login-logo-name{font-size:21px;font-weight:600;letter-spacing:.8px;color:#FFFFFF;line-height:1.1}.login-logo-sub{font-size:12px;color:#8FA6B2;margin-top:3px}
.login-eyebrow{font-size:12px;color:#9FB6C2;letter-spacing:.8px;text-transform:uppercase;margin-bottom:10px}.login-title{font-size:33px;line-height:1.15;font-weight:600;letter-spacing:-.5px;color:#FFFFFF;margin-bottom:15px}
.login-copy{max-width:450px;color:#C5D4DB;font-size:14px;line-height:1.65}.login-system{position:absolute;left:42px;right:42px;bottom:30px;border-top:1px solid #26404F;padding-top:16px;color:#8FA6B2;font-size:12px;z-index:2}.login-system strong{color:#E6EEF2;font-weight:500}
.login-form{min-height:500px;padding:48px 44px;background:#FFFFFF;border-radius:0 8px 8px 0}.login-form-title{font-size:25px;font-weight:600;color:#14202B;letter-spacing:-.2px;margin:85px 0 6px 0}.login-form-sub{font-size:13px;color:#6B7A87;margin-bottom:25px}
.login-error{background:#F9E4E1;border-left:3px solid #B4372E;color:#8F2A22;padding:10px 12px;border-radius:0 4px 4px 0;font-size:13px;margin-bottom:13px}
.login-form div[data-testid="stTextInput"] label{color:#5F6E7B !important;font-size:12px !important}.login-form div[data-testid="stTextInput"] input{background:#FFFFFF !important;color:#14202B !important;border:1px solid #D9DFE5 !important;border-radius:5px !important;height:44px !important;font-size:14px !important}.login-form div[data-testid="stTextInput"] input:focus{border-color:#2B6C8F !important;box-shadow:0 0 0 1px #2B6C8F !important}
.login-form .stButton>button{height:44px;margin-top:8px;border-radius:5px;border:1px solid #F2B705;background:#F2B705;color:#12212B;font-weight:600;font-size:14px}.login-form .stButton>button:hover{background:#DFA900;border-color:#DFA900;color:#12212B}
.login-help{font-size:11.5px;line-height:1.5;color:#8794A0;margin-top:15px}.login-demo{font-size:11px;color:#7A8894;margin-top:10px;padding-top:10px;border-top:1px solid #E6EAEE}.login-card{background:#FFFFFF;border:1px solid #D9DFE5;border-radius:8px;box-shadow:0 14px 40px rgba(20,32,43,.12);padding:0;overflow:hidden}
@media(max-width:800px){.login-brand,.login-form{min-height:auto;border-radius:8px;padding:32px 28px}.login-system{position:static;margin-top:42px}.login-title{font-size:28px}.login-form-title{margin-top:0}}
"""
st.markdown(f"<style>{H(LOGIN_CSS)}</style>", unsafe_allow_html=True)


# ============================================================
# ARTWORK  (generated SVG, no external files needed)
# ============================================================

def _contours(parts, cx, cy, r0, step, n, phase, colour, opacity, stretch=1.5):
    for k in range(n):
        r = r0 + k * step
        pts = []
        for i in range(73):
            a = i / 72 * 2 * math.pi
            rr = r * (1 + 0.10 * math.sin(3 * a + phase + k * 0.15)
                      + 0.06 * math.sin(5 * a + phase * 1.7)
                      + 0.03 * math.sin(9 * a + phase * 0.5))
            pts.append(f"{cx + rr * math.cos(a) * stretch:.1f},{cy + rr * math.sin(a):.1f}")
        op = max(opacity - k * 0.006, 0.03)
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colour}" '
            f'stroke-opacity="{op:.3f}" stroke-width="1"/>'
        )


@lru_cache(maxsize=1)
def header_art_uri() -> str:
    """Header background: a real photo if one exists, otherwise generated terrain contours."""
    path = os.path.join("assets", "site_banner.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            "linear-gradient(90deg, rgba(18,33,43,.94) 30%, rgba(18,33,43,.55) 100%), "
            f"url('data:image/jpeg;base64,{b64}')"
        )
    w, h = 1600, 230
    parts = []
    _contours(parts, w * 0.80, h * 0.55, 14, 17, 17, 0.6, "#9DC2D3", 0.28)
    _contours(parts, w * 0.50, h * 1.10, 10, 18, 12, 2.1, "#9DC2D3", 0.20)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMaxYMid slice">'
        f'<rect width="100%" height="100%" fill="#12212B"/>{"".join(parts)}</svg>'
    )
    return f"url('{svg_uri(svg)}')"


LOGO = (
    '<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="34" height="34" rx="7" fill="#F2B705"/>'
    '<polyline points="5,19 11,19 15,10 20,26 24,15 29,15" fill="none" stroke="#12212B" '
    'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

# ============================================================
# LOGIN / AUTHENTICATION
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "login_error" not in st.session_state:
    st.session_state.login_error = ""

if not st.session_state.authenticated:
    st.markdown("<style>section[data-testid='stSidebar']{display:none!important;} .block-container{max-width:1180px!important;padding-top:2rem!important;}</style>", unsafe_allow_html=True)
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    left, right = st.columns([1.08, 0.92], gap="small")
    with left:
        md(f'<div class="login-brand"><div class="login-brand-content"><div class="login-logo-row">{LOGO}<div><div class="login-logo-name">INNOVATRIX</div><div class="login-logo-sub">Mine monitoring console</div></div></div><div class="login-eyebrow">Smart Automation · Hardware Prototype</div><div class="login-title">Real-time mine<br>subsidence monitoring.</div><div class="login-copy">Monitor ground tilt, vibration and relative displacement across distributed sensor nodes, with a combined risk index for early warning.</div></div><div class="login-system"><strong>Prototype Mine</strong> · GW-001 · SX1278 LoRa · 4 monitoring nodes</div></div>')
    with right:
        st.markdown('<div class="login-form"><div class="login-form-title">Sign in</div><div class="login-form-sub">Access the INNOVATRIX monitoring console</div>', unsafe_allow_html=True)
        if st.session_state.login_error:
            st.markdown(f'<div class="login-error">{st.session_state.login_error}</div>', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            if username.strip() == "admin" and password == "innovatrix@2026":
                st.session_state.authenticated = True
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "Invalid username or password. Please check your credentials."
                st.rerun()
        md('<div class="login-help">Prototype access for the project demonstration. Production deployment should use secure authentication and secrets management.</div><div class="login-demo"><b>Demo account:</b> admin</div>')
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ============================================================
# END LOGIN / AUTHENTICATION
# ============================================================


def section_svg(states: dict, tilts: dict) -> str:
    """Schematic cross-section: surface nodes above an extracted coal panel."""
    W, H_ = 900, 388
    tr = lambda x: math.exp(-(((x - 500) / 140) ** 2))
    ground = lambda x: 98 + 15 * tr(x) + 3 * math.sin(x / 65)
    l_soil = lambda x: ground(x) + 20
    l_sand = lambda x: 170 + 9 * tr(x)
    l_shale = lambda x: 240 + 4 * tr(x)
    l_coal = lambda x: 268 + 3 * tr(x)
    floor_y = 356
    xs = list(range(0, W + 1, 10))

    def band(top, bottom, fill, xr=xs):
        pts = [f"{x},{top(x):.1f}" for x in xr] + [f"{x},{bottom(x):.1f}" for x in reversed(xr)]
        return f'<polygon points="{" ".join(pts)}" fill="{fill}"/>'

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H_}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H_}" fill="#E9EEF2"/>']
    o.append(band(ground, l_soil, "#A99F86"))
    o.append(band(l_soil, l_sand, "#CBC5B5"))
    o.append(band(l_sand, l_shale, "#A9B1B5"))
    o.append(band(l_shale, l_coal, "#2A3439"))
    o.append(band(l_coal, lambda x: floor_y, "#8B969B"))

    # extracted panel (void) with remaining pillars
    gx = list(range(330, 701, 10))
    o.append(band(l_shale, l_coal, "#10181D", gx))
    for px in (355, 425, 565, 635):
        o.append(f'<rect x="{px}" y="243" width="18" height="26" fill="#3A464C"/>')

    # angle-of-draw lines
    o.append(f'<line x1="330" y1="240" x2="272" y2="{ground(272):.1f}" stroke="#FFFFFF" stroke-opacity=".7" '
             'stroke-dasharray="6 5" stroke-width="1.2"/>')
    o.append(f'<line x1="700" y1="240" x2="758" y2="{ground(758):.1f}" stroke="#FFFFFF" stroke-opacity=".7" '
             'stroke-dasharray="6 5" stroke-width="1.2"/>')

    # original surface level + actual surface
    o.append('<line x1="0" y1="96" x2="900" y2="96" stroke="#7B8A96" stroke-dasharray="3 4" stroke-width="1"/>')
    o.append(f'<polyline points="{" ".join(f"{x},{ground(x):.1f}" for x in xs)}" fill="none" '
             'stroke="#3E4A39" stroke-width="2"/>')
    o.append('<text x="108" y="90" font-size="11" fill="#5F6E7B">Original surface level</text>')
    o.append('<text x="405" y="156" font-size="11" fill="#4B4535" text-anchor="middle">Subsidence trough</text>')

    # strata labels
    o.append(f'<text x="14" y="{ground(14) + 15:.0f}" font-size="11" fill="#2E2A20">Soil</text>')
    o.append('<text x="14" y="152" font-size="11" fill="#3E3B31">Sandstone</text>')
    o.append('<text x="14" y="214" font-size="11" fill="#2B363D">Shale</text>')
    o.append('<text x="14" y="259" font-size="11" fill="#C9D5DC">Coal seam</text>')
    o.append('<text x="14" y="322" font-size="11" fill="#1F2E38">Floor rock</text>')

    # panel bracket + label
    o.append('<path d="M330 282 V290 H700 V282" fill="none" stroke="#1F2E38" stroke-width="1.2"/>')
    o.append('<text x="515" y="308" font-size="12" fill="#1F2E38" text-anchor="middle">'
             'Extracted panel</text>')

    # gateway mast
    gy = ground(60)
    top = gy - 70
    o.append(f'<line x1="60" y1="{gy:.0f}" x2="60" y2="{top:.0f}" stroke="#1F2E38" stroke-width="3"/>')
    o.append(f'<rect x="50" y="{gy - 10:.0f}" width="20" height="10" fill="#1F2E38"/>')
    o.append(f'<circle cx="60" cy="{top:.0f}" r="4" fill="#F2B705" stroke="#1F2E38" stroke-width="1.5"/>')
    for r in (9, 15):
        o.append(f'<path d="M{60 + r * 0.7:.1f} {top - r * 0.7:.1f} A{r} {r} 0 0 1 {60 + r * 0.7:.1f} {top + r * 0.7:.1f}" '
                 f'fill="none" stroke="{STEEL}" stroke-width="1.6"/>')
    o.append(f'<text x="60" y="{top - 12:.0f}" font-size="12" font-weight="600" fill="{INK}" text-anchor="middle">GW-001</text>')

    # LoRa links, reference lines, nodes
    for nid, nd in NODES.items():
        x = nd["sx"]
        gyn = ground(x)
        o.append(f'<path d="M66 {top:.0f} Q{(66 + x) / 2:.0f} {8 + (x / 900) * 6:.0f} {x} 32" fill="none" '
                 f'stroke="{STEEL}" stroke-opacity=".5" stroke-dasharray="5 4" stroke-width="1.2"/>')
        o.append(f'<line x1="{x}" y1="{gyn:.0f}" x2="{x}" y2="240" stroke="#FFFFFF" stroke-opacity=".55" '
                 'stroke-dasharray="2 4" stroke-width="1.2"/>')
    for nid, nd in NODES.items():
        x = nd["sx"]
        gyn = ground(x)
        c = STATUS[states[nid]]["color"]
        o.append(f'<line x1="{x}" y1="56" x2="{x}" y2="{gyn - 8:.0f}" stroke="{INK}" stroke-width="1.5"/>')
        o.append(f'<rect x="{x - 52}" y="30" width="104" height="26" rx="4" fill="#FFFFFF" stroke="{c}" stroke-width="1.6"/>')
        o.append(f'<text x="{x - 44}" y="47" font-size="12" font-weight="600" fill="{INK}">{nid}</text>')
        o.append(f'<text x="{x + 46}" y="47" font-size="12" fill="{INK}" text-anchor="end">{tilts[nid]:.2f}°</text>')
        o.append(f'<circle cx="{x}" cy="{gyn - 6:.0f}" r="8" fill="{c}" stroke="#FFFFFF" stroke-width="2.5"/>')

    # depth scale
    o.append(f'<line x1="884" y1="96" x2="884" y2="268" stroke="#F2F2F2" stroke-width="1.2"/>')
    for yy in (96, 268):
        o.append(f'<line x1="879" y1="{yy}" x2="889" y2="{yy}" stroke="#F2F2F2" stroke-width="1.2"/>')
    o.append('<text x="876" y="186" font-size="11" fill="#F2F2F2" text-anchor="end">Depth about 170 m</text>')
    o.append('</svg>')
    return "".join(o)


def plan_svg(states: dict, tilts: dict) -> str:
    """Plan view: gateway, LoRa links and nodes above the panel footprint."""
    W, H_ = 900, 470
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H_}" font-family="{FONT}">',
         '<defs><pattern id="hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<line x1="0" y1="0" x2="0" y2="9" stroke="#B7791F" stroke-opacity=".35" stroke-width="2"/></pattern></defs>',
         f'<rect width="{W}" height="{H_}" fill="#EDF1EE"/>']
    parts = []
    _contours(parts, 700, 130, 20, 26, 12, 1.1, "#9FB0AA", 0.55, stretch=1.35)
    _contours(parts, 190, 420, 16, 24, 9, 3.0, "#9FB0AA", 0.5, stretch=1.35)
    o += parts

    # haulage road
    road = "0,385 200,352 400,398 650,382 900,332"
    o.append(f'<polyline points="{road}" fill="none" stroke="#C4CDD3" stroke-width="15" stroke-linejoin="round"/>')
    o.append(f'<polyline points="{road}" fill="none" stroke="#FFFFFF" stroke-width="10" stroke-linejoin="round"/>')
    o.append('<text x="420" y="424" font-size="11" fill="#6B7A87">Haulage road</text>')

    # panel footprint
    o.append('<polygon points="330,150 700,128 735,300 360,332" fill="url(#hatch)" stroke="#7B6A45" '
             'stroke-width="1.4" stroke-dasharray="7 5"/>')
    o.append('<text x="545" y="232" font-size="12" fill="#5B4B2A" text-anchor="middle">Extracted panel footprint</text>')

    gx, gy = GATEWAY_MAP
    for nid, nd in NODES.items():
        o.append(f'<line x1="{gx}" y1="{gy}" x2="{nd["mx"]}" y2="{nd["my"]}" stroke="{STEEL}" '
                 'stroke-opacity=".55" stroke-dasharray="6 5" stroke-width="1.3"/>')
        d = math.hypot(nd["mx"] - gx, nd["my"] - gy) * 0.5
        lx, ly = (gx + nd["mx"]) / 2, (gy + nd["my"]) / 2
        o.append(f'<rect x="{lx - 21:.0f}" y="{ly - 9:.0f}" width="42" height="16" rx="3" fill="#EDF1EE"/>')
        o.append(f'<text x="{lx:.0f}" y="{ly + 3:.0f}" font-size="10.5" fill="{STEEL}" text-anchor="middle">{d:.0f} m</text>')

    o.append(f'<rect x="{gx - 12}" y="{gy - 12}" width="24" height="24" rx="3" fill="{INK}"/>')
    o.append(f'<circle cx="{gx}" cy="{gy}" r="4" fill="#F2B705"/>')
    o.append(f'<text x="{gx}" y="{gy + 30}" font-size="12" font-weight="600" fill="{INK}" text-anchor="middle">GW-001</text>')

    for nid, nd in NODES.items():
        c = STATUS[states[nid]]["color"]
        x, y = nd["mx"], nd["my"]
        o.append(f'<circle cx="{x}" cy="{y}" r="12" fill="{c}" stroke="#FFFFFF" stroke-width="3"/>')
        lx = x + 16 if x < 700 else x - 128  # flip the label to the left near the right edge
        o.append(f'<rect x="{lx}" y="{y - 22}" width="112" height="40" rx="4" fill="#FFFFFF" stroke="{LINE}"/>')
        o.append(f'<text x="{lx + 8}" y="{y - 6}" font-size="12" font-weight="600" fill="{INK}">{nid}</text>')
        o.append(f'<text x="{lx + 106}" y="{y - 6}" font-size="12" fill="{INK}" text-anchor="end">{tilts[nid]:.2f}°</text>')
        o.append(f'<text x="{lx + 8}" y="{y + 10}" font-size="11" fill="{MUTED}">{nd["zone"]}</text>')

    # scale bar + north arrow
    o.append(f'<line x1="24" y1="446" x2="224" y2="446" stroke="{INK}" stroke-width="2"/>')
    for xx in (24, 124, 224):
        o.append(f'<line x1="{xx}" y1="440" x2="{xx}" y2="452" stroke="{INK}" stroke-width="2"/>')
    o.append(f'<text x="24" y="434" font-size="11" fill="{INK}">0</text>')
    o.append(f'<text x="224" y="434" font-size="11" fill="{INK}" text-anchor="end">100 m</text>')
    o.append(f'<polygon points="868,30 858,62 868,55 878,62" fill="{INK}"/>')
    o.append(f'<text x="868" y="24" font-size="12" font-weight="600" fill="{INK}" text-anchor="middle">N</text>')
    o.append('</svg>')
    return "".join(o)


def img(svg: str, alt: str) -> str:
    return f'<img class="art" src="{svg_uri(svg)}" alt="{alt}"/>'


LEGEND = (
    '<div class="legend">'
    f'<span><i style="background:{STATUS["Normal"]["color"]}"></i>Normal</span>'
    f'<span><i style="background:{STATUS["Watch"]["color"]}"></i>Watch</span>'
    f'<span><i style="background:{STATUS["Warning"]["color"]}"></i>Warning</span>'
    '<span class="lg-note">Node labels show current tilt. Schematic, not to scale.</span></div>'
)


# ============================================================
# SIMULATED SENSOR DATA
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def build_data(minute_key: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 360
    end = datetime.now().replace(second=0, microsecond=0)
    times = [end - timedelta(minutes=n - 1 - i) for i in range(n)]
    frames = []
    for nid, nd in NODES.items():
        d_tilt, d_vib, d_dist = nd["drift"]

        walk = np.cumsum(rng.normal(0, 0.012, n))
        walk -= walk[-1]
        tilt = nd["tilt"] + walk + np.linspace(-d_tilt, 0, n) + rng.normal(0, 0.006, n)

        vib = nd["vib"] + rng.normal(0, 1.1, n) + np.linspace(-d_vib, 0, n)
        vib += (rng.random(n) < 0.012) * rng.uniform(3, 6, n)

        walk2 = np.cumsum(rng.normal(0, 0.02, n))
        walk2 -= walk2[-1]
        dist = nd["dist"] + walk2 + np.linspace(-d_dist, 0, n) + rng.normal(0, 0.008, n)

        tilt[-1], vib[-1], dist[-1] = nd["tilt"], nd["vib"], nd["dist"]  # newest sample = target

        frames.append(pd.DataFrame({
            "time": times,
            "node": nid,
            "tilt": np.clip(tilt, 1.45, 2.9),
            "vibration": np.clip(vib, 0, 20),
            "distance": np.clip(dist, 11.3, 13.9),
        }))
    df = pd.concat(frames, ignore_index=True)

    # same scoring logic as the original prototype
    df["r_tilt"] = np.clip(abs(df["tilt"] - TILT_BASE) / 1.2 * 45, 0, 45)
    df["r_vib"] = np.clip(df["vibration"] / 20 * 25, 0, 25)
    df["r_move"] = np.clip(abs(df["distance"] - DIST_REF) / 2.0 * 30, 0, 30)
    df["risk"] = np.clip(df["r_tilt"] + df["r_vib"] + df["r_move"], 0, 100)
    return df


df = build_data(datetime.now().strftime("%Y%m%d%H%M"))
latest = df.groupby("node").tail(1).set_index("node")
latest["state"] = latest["risk"].apply(state_of)

site_node = latest["risk"].idxmax()
site_risk = float(latest.loc[site_node, "risk"])
site_state = state_of(site_risk)
now_t = latest["time"].max()

states = latest["state"].to_dict()
tilts = latest["tilt"].to_dict()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    md(f"""
    <div class="brand">{LOGO}
      <div><div class="brand-name">INNOVATRIX</div><div class="brand-sub">Mine monitoring console</div></div>
    </div>""")

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    md('<div class="side-h">Time range</div>')
    window = st.selectbox("Time range", list(WINDOWS), index=1, label_visibility="collapsed")

    md('<div class="side-h">Site</div>')
    for k, v in [("Name", "Prototype Mine"), ("Gateway", "GW-001"),
                 ("Radio", "SX1278 LoRa"), ("Nodes", f"{len(NODES)} configured")]:
        md(f'<div class="side-kv"><span>{k}</span><span>{v}</span></div>')

    md('<div class="side-h">System</div>')
    md(f"""<div class="side-state"><span class="dot" style="background:{STATUS['Normal']['color']}"></span>
    <div>Gateway online<small>Simulated sensor data</small></div></div>""")
    st.write("")
    if st.button("Refresh data", use_container_width=True):
        build_data.clear()
        st.rerun()
    st.write("")
    if st.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.login_error = ""
        st.rerun()

n_min = WINDOWS[window]
view = df[df["time"] >= df["time"].max() - timedelta(minutes=n_min - 1)]


# ============================================================
# PAGE HEADER
# ============================================================

title, desc = PAGE_COPY[page]
md(f"""
<div class="pagehead" style="background-image:{header_art_uri()}">
  <div>
    <div class="crumb">Prototype Mine / {page}</div>
    <div class="pagetitle">{title}</div>
    <p>{desc}</p>
  </div>
  <div class="headstat">
    <div class="hs-state"><span class="dot" style="background:{STATUS[site_state]['color']}"></span>{site_state}</div>
    <div class="hs-line">Site risk index {site_risk:.1f} of 100, highest at {site_node}</div>
    <div class="hs-line">Updated {datetime.now():%H:%M:%S}</div>
  </div>
</div>""")


# ============================================================
# CHART BUILDERS
# ============================================================

def base_layout(fig, height, legend=True, top=8):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=top + (30 if legend else 0), b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color="#4B5A67"),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=GRID, zeroline=False, showline=True, linecolor=LINE, tickformat="%H:%M"),
        yaxis=dict(gridcolor=GRID, zeroline=False),
    )
    return fig


def limit_lines(fig, limits):
    for val, name in zip(limits, ("Watch level", "Warning level")):
        colour = STATUS["Watch" if name.startswith("Watch") else "Warning"]["color"]
        fig.add_hline(y=val, line=dict(color=colour, width=1, dash="dot"),
                      annotation_text=name, annotation_position="top left",
                      annotation_font=dict(size=11, color=colour))


def trend_fig(data, col, nodes, ylabel, limits=None, height=340):
    fig = go.Figure()
    for nid in nodes:
        d = data[data["node"] == nid]
        fig.add_trace(go.Scatter(x=d["time"], y=d[col], mode="lines", name=nid,
                                 line=dict(color=NODES[nid]["color"], width=2),
                                 hovertemplate="%{y:.2f}"))
    if limits:
        limit_lines(fig, limits)
    base_layout(fig, height)
    fig.update_yaxes(title_text=ylabel)
    if limits and len(nodes):
        vals = data[data["node"].isin(nodes)][col]
        fig.update_yaxes(range=[min(vals.min(), limits[0]) * 0.97, max(vals.max(), limits[1]) * 1.02])
    return fig


def contrib_fig(lat):
    ids = list(NODES)[::-1]
    fig = go.Figure()
    for key, label, colour in [("r_tilt", "Tilt", "#2B6C8F"),
                               ("r_vib", "Vibration", "#94B6C6"),
                               ("r_move", "Displacement", "#C9A45C")]:
        fig.add_trace(go.Bar(y=ids, x=[lat.loc[i, key] for i in ids], name=label, orientation="h",
                             marker_color=colour, hovertemplate="%{x:.1f} points"))
    for i in ids:
        fig.add_annotation(x=lat.loc[i, "risk"], y=i, text=f"{lat.loc[i, 'risk']:.0f}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=12, color=INK))
    for v, c in ((WATCH_AT, STATUS["Watch"]["color"]), (WARNING_AT, STATUS["Warning"]["color"])):
        fig.add_vline(x=v, line=dict(color=c, width=1, dash="dot"))
    base_layout(fig, 290)
    fig.update_layout(barmode="stack", hovermode="closest", bargap=0.45)
    fig.update_xaxes(range=[0, 100], tickformat=None, title_text="Risk points")
    return fig


def gauge_fig(v):
    s = state_of(v)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        number={"font": {"size": 44, "family": FONT, "color": INK}, "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, 35, 60, 100], "tickfont": {"size": 11}},
               "bar": {"color": STATUS[s]["color"], "thickness": 0.3},
               "bgcolor": "white", "borderwidth": 0,
               "steps": [{"range": [0, WATCH_AT], "color": STATUS["Normal"]["soft"]},
                         {"range": [WATCH_AT, WARNING_AT], "color": STATUS["Watch"]["soft"]},
                         {"range": [WARNING_AT, 100], "color": STATUS["Warning"]["soft"]}]},
    ))
    fig.update_layout(height=230, margin=dict(l=26, r=26, t=16, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT))
    return fig


def risk_history_fig(data):
    hist = data.groupby("time")["risk"].max().reset_index()
    fig = go.Figure()
    for lo, hi, s in ((0, WATCH_AT, "Normal"), (WATCH_AT, WARNING_AT, "Watch"), (WARNING_AT, 100, "Warning")):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=STATUS[s]["soft"], line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["risk"], mode="lines", name="Site risk index",
                             line=dict(color=INK, width=2), hovertemplate="%{y:.1f}"))
    base_layout(fig, 260, legend=False)
    fig.update_yaxes(range=[0, 100], title_text="Risk index")
    return fig


def spark_fig(data, col, colour, limits, height=190):
    fig = go.Figure(go.Scatter(x=data["time"], y=data[col], mode="lines",
                               line=dict(color=colour, width=2), hovertemplate="%{y:.2f}"))
    limit_lines(fig, limits)
    base_layout(fig, height, legend=False)
    vals = data[col]
    fig.update_yaxes(range=[min(vals.min(), limits[0]) * 0.96, max(vals.max(), limits[1]) * 1.02])
    return fig


# ============================================================
# TABLE BUILDERS
# ============================================================

def signal_bars(rssi):
    n = 4 if rssi >= -75 else 3 if rssi >= -90 else 2 if rssi >= -105 else 1
    return '<span class="sig">' + "".join(f'<i class="{"on" if i < n else ""}"></i>' for i in range(4)) + "</span>"


def battery_bar(p):
    c = STATUS["Normal"]["color"] if p >= 50 else STATUS["Watch"]["color"] if p >= 20 else STATUS["Warning"]["color"]
    return f'<span class="bat"><b style="width:{p}%;background:{c}"></b></span>{p}%'


def node_table(lat):
    rows = ""
    for nid, nd in NODES.items():
        r = lat.loc[nid]
        rows += (
            f'<tr><td><b>{nid}</b><span class="sub">{nd["zone"]}, {nd["where"]}</span></td>'
            f'<td>{pill(r["state"])}</td><td>{battery_bar(nd["batt"])}</td>'
            f'<td>{signal_bars(nd["rssi"])}{nd["rssi"]} dBm</td>'
            f'<td class="num">{r["tilt"]:.2f}°</td><td class="num">{r["vibration"]:.1f}</td>'
            f'<td class="num">{r["distance"]:.2f} mm</td><td class="num">{r["risk"]:.1f}</td>'
            f'<td class="num">{nd["seen"]} s ago</td></tr>'
        )
    return (
        '<table class="dt"><thead><tr><th>Node</th><th>State</th><th>Battery</th><th>LoRa signal</th>'
        '<th class="num">Tilt</th><th class="num">Vibration</th><th class="num">Distance</th>'
        '<th class="num">Risk</th><th class="num">Last packet</th></tr></thead><tbody>'
        f'{rows}</tbody></table>'
    )


def make_events(lat, t):
    n3 = lat.loc["N-003"]
    n4 = NODES["N-004"]
    items = [
        (2, "N-003", "Watch", f"Tilt is {n3['tilt']:.2f}°, {n3['tilt'] - TILT_BASE:.2f}° above the {TILT_BASE:.1f}° baseline (watch level {LIMITS['tilt'][0]:.1f}°)."),
        (9, "N-003", "Watch", f"Distance reading {n3['distance']:.1f} mm, {n3['distance'] - DIST_REF:+.1f} mm from the fixed reference."),
        (17, "N-002", "Info", "Vibration activity rose compared with the previous interval. No level crossed."),
        (26, "GW-001", "Info", f"Gateway heartbeat received. {len(NODES)} of {len(NODES)} nodes reporting."),
        (41, "N-004", "Info", f"Routine node health update received. Battery {n4['batt']}%."),
        (58, "N-001", "Info", "Sensor data received and stored."),
        (75, "N-003", "Watch", f"Site risk index passed {WATCH_AT} and moved from Normal to Watch."),
        (120, "N-004", "Info", "Battery fell below 80%. Schedule a replacement at the next inspection."),
    ]
    return [(t - timedelta(minutes=m), m, node, sev, msg) for m, node, sev, msg in items]


def events_html(events):
    out = ""
    for ts, m, node, sev, msg in events:
        ago = f"{m} min ago" if m < 60 else f"{m // 60} h {m % 60:02d} min ago"
        out += (f'<div class="ev"><div class="ev-t">{ts:%H:%M}<small>{ago}</small></div>'
                f'<div class="ev-n">{node}</div><div class="ev-m">{msg}</div><div>{pill(sev)}</div></div>')
    return out


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "Overview":
    lat = latest
    online = int((lat.index.map(lambda n: NODES[n]["seen"]) < 120).sum())
    low_node = min(NODES, key=lambda n: NODES[n]["batt"])
    t_node = lat["tilt"].idxmax()
    d_node = (lat["distance"] - DIST_REF).abs().idxmax()
    v_node = lat["vibration"].idxmax()
    t_val = lat.loc[t_node, "tilt"]
    d_val = lat.loc[d_node, "distance"] - DIST_REF
    v_val = lat.loc[v_node, "vibration"]
    green = STATUS["Normal"]["color"]

    md(f"""
    <div class="kpis">
      <div class="kpi"><div class="kpi-l">Site risk index</div>
        <div class="kpi-v">{site_risk:.1f}<small>of 100</small></div>
        {meter(site_risk, WATCH_AT, WARNING_AT, 100)}
        <div class="kpi-n">{site_state}, highest at {site_node}</div></div>
      <div class="kpi"><div class="kpi-l">Nodes online</div>
        <div class="kpi-v">{online}<small>of {len(NODES)}</small></div>
        <div class="meter"><div class="meter-fill" style="width:{online / len(NODES) * 100:.0f}%;background:{green}"></div></div>
        <div class="kpi-n">Lowest battery {NODES[low_node]['batt']}% at {low_node}</div></div>
      <div class="kpi"><div class="kpi-l">Highest tilt</div>
        <div class="kpi-v">{t_val:.2f}<small>degrees</small></div>
        {meter(t_val, *LIMITS['tilt'], 3.2)}
        <div class="kpi-n">{t_node}, {t_val - TILT_BASE:+.2f}° from baseline</div></div>
      <div class="kpi"><div class="kpi-l">Largest displacement</div>
        <div class="kpi-v">{d_val:+.1f}<small>mm</small></div>
        {meter(abs(d_val), LIMITS['distance'][0] - DIST_REF, LIMITS['distance'][1] - DIST_REF, 2.4)}
        <div class="kpi-n">{d_node}, from {DIST_REF:.1f} mm reference</div></div>
      <div class="kpi"><div class="kpi-l">Peak vibration</div>
        <div class="kpi-v">{v_val:.1f}<small>index</small></div>
        {meter(v_val, *LIMITS['vibration'], 20)}
        <div class="kpi-n">{v_node}, scale 0 to 20</div></div>
    </div>""")

    left, right = st.columns([2.2, 1])
    with left:
        with panel("Site cross-section", "Nodes at their installed positions"):
            md(img(section_svg(states, tilts), "Schematic cross-section of the mine site with sensor nodes on the surface"))
            md(LEGEND)
    with right:
        with panel("Node risk index", f"Watch from {WATCH_AT}, warning from {WARNING_AT}"):
            rows = ""
            for nid, nd in NODES.items():
                r = lat.loc[nid]
                rows += (
                    f'<div class="nr"><div class="nr-top"><span><span class="nr-id">{nid}</span>'
                    f'<span class="nr-z">{nd["zone"]}</span></span>{pill(r["state"])}</div>'
                    f'{meter(r["risk"], WATCH_AT, WARNING_AT, 100)}'
                    f'<div class="nr-foot"><span>Tilt {r["tilt"]:.2f}°, vibration {r["vibration"]:.1f}, '
                    f'displacement {r["distance"] - DIST_REF:+.1f} mm</span><b>{r["risk"]:.1f}</b></div></div>'
                )
            md(rows)

    a, b = st.columns([1.8, 1])
    with a:
        with panel("Ground tilt by node", window):
            show(trend_fig(view, "tilt", list(NODES), "Tilt (°)", LIMITS["tilt"]), "ov_tilt")
    with b:
        with panel("What drives each node's score", "Points out of 100"):
            show(contrib_fig(lat), "ov_contrib")

    g, h = st.columns([1, 2])
    with g:
        with panel("Site risk index", "Highest node score"):
            show(gauge_fig(site_risk), "ov_gauge")
            md(f'<div style="text-align:center">{pill(site_state)}</div>')
    with h:
        with panel("Risk index history", window):
            show(risk_history_fig(view), "ov_hist")

    with panel("Recent events", "Latest four"):
        md(events_html(make_events(latest, now_t)[:4]))

    md("""<div class="notice">Risk scores and levels on this console are prototype values. They must be calibrated
    and validated against representative mine sensor data, and approved under the mine's safety procedures,
    before they are used operationally.</div>""")


# ============================================================
# PAGE: SENSOR TRENDS
# ============================================================

elif page == "Sensor trends":
    PARAMS = {
        "Tilt": ("tilt", "Tilt (°)", LIMITS["tilt"]),
        "Vibration": ("vibration", "Vibration index", LIMITS["vibration"]),
        "Relative displacement": ("distance", "Distance (mm)", LIMITS["distance"]),
        "Risk score": ("risk", "Risk points", (WATCH_AT, WARNING_AT)),
    }
    c1, c2 = st.columns([1, 2])
    with c1:
        param = st.selectbox("Parameter", list(PARAMS))
    with c2:
        sel = st.multiselect("Nodes", list(NODES), default=list(NODES))
    col, ylabel, lims = PARAMS[param]

    if not sel:
        md('<div class="notice">Select at least one node to see its readings.</div>')
    else:
        with panel(f"{param} by node", window):
            show(trend_fig(view, col, sel, ylabel, lims, height=400), "tr_main")

        a, b = st.columns([1, 1])
        with a:
            with panel("Summary for the selected range", ylabel):
                stats = (view[view["node"].isin(sel)].groupby("node")[col]
                         .agg(Latest="last", Minimum="min", Average="mean", Maximum="max", Spread="std")
                         .round(2).reset_index().rename(columns={"node": "Node"}))
                st.dataframe(stats, use_container_width=True, hide_index=True)
        with b:
            with panel("Distribution of readings", "Count of samples"):
                fig = go.Figure()
                for nid in sel:
                    d = view[view["node"] == nid]
                    fig.add_trace(go.Histogram(x=d[col], name=nid, nbinsx=24, opacity=0.7,
                                               marker_color=NODES[nid]["color"]))
                base_layout(fig, 250)
                fig.update_layout(barmode="overlay", hovermode="closest")
                fig.update_xaxes(title_text=ylabel, tickformat=None)
                show(fig, "tr_hist")

        with panel("Recent sensor records", "Newest first"):
            rec = (view[view["node"].isin(sel)].sort_values("time", ascending=False).head(12)
                   [["time", "node", "tilt", "vibration", "distance", "risk"]].copy())
            rec["time"] = rec["time"].dt.strftime("%H:%M")
            rec.columns = ["Time", "Node", "Tilt (°)", "Vibration", "Distance (mm)", "Risk"]
            st.dataframe(rec.round(2), use_container_width=True, hide_index=True)
            st.download_button("Download range as CSV",
                               data=view[view["node"].isin(sel)].drop(columns=["r_tilt", "r_vib", "r_move"])
                               .to_csv(index=False),
                               file_name="sensor_readings.csv", mime="text/csv")


# ============================================================
# PAGE: SENSOR NODES
# ============================================================

elif page == "Sensor nodes":
    with panel("Node health", "Live from the LoRa gateway"):
        md(node_table(latest))

    sel = st.selectbox("Node detail", list(NODES), format_func=lambda n: f"{n}, {NODES[n]['zone']} ({NODES[n]['where']})")
    nd = NODES[sel]
    r = latest.loc[sel]
    d = view[view["node"] == sel]

    s1, s2, s3 = st.columns(3)
    with s1:
        with panel("Tilt, MPU6050", f"{r['tilt']:.2f}° now"):
            show(spark_fig(d, "tilt", nd["color"], LIMITS["tilt"]), "nd_tilt")
    with s2:
        with panel("Vibration, SW-420", f"{r['vibration']:.1f} now"):
            show(spark_fig(d, "vibration", nd["color"], LIMITS["vibration"]), "nd_vib")
    with s3:
        with panel("Distance, VL53L0X", f"{r['distance']:.2f} mm now"):
            show(spark_fig(d, "distance", nd["color"], LIMITS["distance"]), "nd_dist")

    k1, k2 = st.columns(2)
    with k1:
        with panel("Hardware", sel):
            md(f"""<div class="kv">
              <div>Controller</div><div>ESP32</div>
              <div>Radio</div><div>SX1278 LoRa</div>
              <div>Tilt sensor</div><div>MPU6050</div>
              <div>Vibration sensor</div><div>SW-420</div>
              <div>Distance sensor</div><div>VL53L0X</div>
              <div>Firmware</div><div>{nd['fw']}</div></div>""")
    with k2:
        with panel("Link, power and service", sel):
            md(f"""<div class="kv">
              <div>State</div><div>{pill(r['state'])}</div>
              <div>Risk score</div><div>{r['risk']:.1f} of 100</div>
              <div>Battery</div><div>{battery_bar(nd['batt'])}</div>
              <div>Signal</div><div>{signal_bars(nd['rssi'])}{nd['rssi']} dBm</div>
              <div>Last packet</div><div>{nd['seen']} s ago</div>
              <div>Installed</div><div>{nd['installed']}</div>
              <div>Last calibration</div><div>{nd['calibrated']}</div></div>""")

    md("""<div class="notice">Node health values are simulated. In the hardware version these fields are filled
    from ESP32 packets received through the LoRa gateway.</div>""")


# ============================================================
# PAGE: SITE MAP
# ============================================================

elif page == "Site map":
    with panel("Plan view", "Gateway, LoRa links and nodes"):
        md(img(plan_svg(states, tilts), "Plan view of the site showing the gateway, nodes and extracted panel footprint"))
        md(LEGEND.replace("Schematic, not to scale.", "Link labels show approximate distance to the gateway."))

    with panel("Node positions", "Distances measured on the plan"):
        gx, gy = GATEWAY_MAP
        rows = ""
        for nid, nd in NODES.items():
            dist_m = math.hypot(nd["mx"] - gx, nd["my"] - gy) * 0.5
            rows += (f'<tr><td><b>{nid}</b></td><td>{nd["zone"]}</td><td>{nd["where"]}</td>'
                     f'<td>{pill(latest.loc[nid, "state"])}</td>'
                     f'<td class="num">{dist_m:.0f} m</td>'
                     f'<td>{signal_bars(nd["rssi"])}{nd["rssi"]} dBm</td></tr>')
        md(f"""<table class="dt"><thead><tr><th>Node</th><th>Zone</th><th>Location</th><th>State</th>
        <th class="num">Distance to gateway</th><th>LoRa signal</th></tr></thead><tbody>{rows}</tbody></table>""")

    md("""<div class="notice">Positions are illustrative for the prototype. Replace them with surveyed coordinates
    when nodes are installed underground and on the surface.</div>""")


# ============================================================
# PAGE: ALERTS AND EVENTS
# ============================================================

else:
    sev = st.multiselect("Severity", ["Warning", "Watch", "Info"], default=["Warning", "Watch", "Info"])
    events = [e for e in make_events(latest, now_t) if e[3] in sev]

    with panel("Event log", f"{len(events)} shown"):
        if events:
            md(events_html(events))
        else:
            md('<div class="notice">No events match the selected severities. Choose more severities to see them.</div>')

    a, b = st.columns([1, 1.3])
    with a:
        with panel("Risk classification", "Applied to the site risk index"):
            cur = site_state
            md(f"""<table class="dt"><thead><tr><th>State</th><th>Risk index</th><th>Current</th></tr></thead><tbody>
            <tr><td>{pill('Normal')}</td><td>Below {WATCH_AT}</td><td>{'Site now' if cur == 'Normal' else ''}</td></tr>
            <tr><td>{pill('Watch')}</td><td>{WATCH_AT} to {WARNING_AT}</td><td>{'Site now' if cur == 'Watch' else ''}</td></tr>
            <tr><td>{pill('Warning')}</td><td>{WARNING_AT} and above</td><td>{'Site now' if cur == 'Warning' else ''}</td></tr>
            </tbody></table>""")
            md(f'<div class="nr-foot"><span>Current site index at {site_node}</span><b>{site_risk:.1f}</b></div>')
    with b:
        with panel("How the score is built", "Points add up to 100"):
            md(f"""<table class="dt"><thead><tr><th>Input</th><th class="num">Max points</th><th>Basis</th></tr></thead><tbody>
            <tr><td>Tilt</td><td class="num">45</td><td>Deviation from the {TILT_BASE:.1f}° baseline, full at 1.2°</td></tr>
            <tr><td>Vibration</td><td class="num">25</td><td>Activity index from 0 to 20</td></tr>
            <tr><td>Displacement</td><td class="num">30</td><td>Change from the {DIST_REF:.1f} mm reference, full at 2.0 mm</td></tr>
            </tbody></table>""")

    md("""<div class="notice">Warning levels are for demonstration. Operational alert thresholds need calibration,
    field validation and approval under the mine's safety procedures.</div>""")


md('<div class="foot">Innovatrix mine subsidence monitoring. Prototype console using simulated sensor data.</div>')
