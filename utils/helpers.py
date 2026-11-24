"""
HealthSense UI Helpers and Visual Utilities.
Provides clinical-grade styling, custom dark sidebar CSS, metric cards, and Plotly gauges.
Matches DiaMetrics design tokens.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from utils.config import THEME, MEDICAL_DISCLAIMER, RISK_CATEGORIES


def inject_custom_css():
    """Inject DiaMetrics modern clinical dashboard CSS into Streamlit."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {THEME['text']};
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.02em;
    }}

    .font-mono {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Canvas Background */
    .stApp {{
        background-color: {THEME['neutral_bg']};
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }}

    /* Dark Modern Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {THEME['sidebar_bg']} !important;
        border-right: 1px solid {THEME['sidebar_border']};
    }}

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] span {{
        color: {THEME['sidebar_text']};
    }}

    /* Sidebar Radio Navigation Items */
    section[data-testid="stSidebar"] .stRadio > div {{
        gap: 4px;
    }}

    section[data-testid="stSidebar"] .stRadio label {{
        background-color: transparent;
        padding: 8px 12px;
        border-radius: 8px;
        color: {THEME['sidebar_text']} !important;
        font-size: 0.88rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
        border: 1px solid transparent;
        margin: 1px 0;
    }}

    section[data-testid="stSidebar"] .stRadio label:hover {{
        background-color: rgba(30, 41, 59, 0.7);
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio div[aria-checked="true"] {{
        background-color: {THEME['sidebar_active']} !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }}

    /* Top Sticky Medical Warning Alert Banner */
    .diametrics-alert-banner {{
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 0.75rem 1.1rem;
        color: #92400E;
        font-size: 0.82rem;
        margin-bottom: 1.5rem;
        line-height: 1.45;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Breadcrumb / Screen Label */
    .screen-breadcrumb {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {THEME['primary']};
        margin-bottom: 0.25rem;
    }}

    /* Main Page Headings */
    .page-title {{
        font-size: 1.95rem;
        font-weight: 700;
        color: {THEME['secondary']};
        margin: 0;
        letter-spacing: -0.03em;
    }}
    .page-subtitle {{
        color: {THEME['muted']};
        font-size: 0.92rem;
        margin-top: 0.3rem;
        margin-bottom: 1.5rem;
    }}

    /* Clean White Cards */
    .diametrics-card {{
        background: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
    }}

    .diametrics-card:hover {{
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }}

    /* KPI Metric Cards */
    .kpi-card {{
        background: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .kpi-label {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {THEME['muted']};
        margin-bottom: 0.35rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: {THEME['secondary']};
        line-height: 1.15;
    }}
    .kpi-subtext {{
        font-size: 0.78rem;
        color: {THEME['muted']};
        margin-top: 0.4rem;
    }}

    /* Action Navigation Cards on Home */
    .action-nav-card {{
        background: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.3rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }}
    .action-nav-card:hover {{
        transform: translateY(-3px);
        border-color: {THEME['primary']};
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
    }}

    /* Clinical Badges */
    .badge-pill {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .badge-low {{
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }}
    .badge-moderate {{
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }}
    .badge-high {{
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }}

    /* Insight Banner */
    .insight-box {{
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid {THEME['primary']};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }}
    .insight-box h4 {{
        color: #1E40AF;
        margin: 0 0 0.35rem 0;
        font-size: 0.98rem;
        font-weight: 600;
    }}
    .insight-box p {{
        color: #1E293B;
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }}

    /* Workflow Pipeline Stepper */
    .workflow-step {{
        background: #FFFFFF;
        border: 1px solid {THEME['border']};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        position: relative;
    }}
    .workflow-step-num {{
        display: inline-block;
        width: 26px;
        height: 26px;
        background: {THEME['primary']};
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 26px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    .workflow-step-title {{
        font-weight: 600;
        font-size: 0.9rem;
        color: {THEME['secondary']};
        margin-bottom: 0.2rem;
    }}
    .workflow-step-desc {{
        font-size: 0.78rem;
        color: {THEME['muted']};
        line-height: 1.35;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_page_header(screen_label: str, title: str, subtitle: str, badge: Optional[str] = None):
    """Render a clean DiaMetrics screen header with breadcrumb and subtitle."""
    badge_html = f"<span class='badge-pill badge-low' style='margin-left: 10px; font-size: 0.7rem;'>{badge}</span>" if badge else ""
    st.markdown(f"""
    <div style='margin-bottom: 1.2rem;'>
        <div class="screen-breadcrumb">{screen_label}</div>
        <h1 class="page-title">{title}{badge_html}</h1>
        <div class="page-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    """Render the top amber warning alert bar."""
    st.markdown(f"""
    <div class="diametrics-alert-banner">
        <div>{MEDICAL_DISCLAIMER}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi(col, label: str, value: Any, subtext: str = "", icon: str = ""):
    """Render a modern clinical KPI card with monospace figures."""
    with col:
        icon_html = f"<span>{icon}</span>" if icon else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">
                <span>{label}</span>
                {icon_html}
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """, unsafe_allow_html=True)


def create_risk_gauge(probability: float) -> go.Figure:
    """Create a clean semi-circular risk gauge matching DiaMetrics."""
    prob_pct = round(probability * 100, 1)

    if prob_pct < 35:
        bar_color = THEME['success']
    elif prob_pct < 65:
        bar_color = THEME['warning']
    else:
        bar_color = THEME['danger']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={'suffix': "%", 'font': {'size': 42, 'family': 'JetBrains Mono', 'color': bar_color, 'weight': 'bold'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Diabetes Risk Probability", 'font': {'size': 14, 'family': 'Outfit', 'color': THEME['secondary']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': THEME['border'], 'tickfont': {'family': 'JetBrains Mono', 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': THEME['border'],
            'steps': [
                {'range': [0, 35], 'color': '#ECFDF5'},
                {'range': [35, 65], 'color': '#FFFBEB'},
                {'range': [65, 100], 'color': '#FEF2F2'}
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 4},
                'thickness': 0.8,
                'value': prob_pct
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans")
    )
    return fig


def create_quality_score_gauge(score: float) -> go.Figure:
    """Create a circular radial gauge for Data Quality Score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "/100", 'font': {'size': 38, 'family': 'JetBrains Mono', 'color': THEME['primary'], 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': THEME['border']},
            'bar': {'color': THEME['primary'], 'thickness': 0.25},
            'steps': [
                {'range': [0, 60], 'color': "#FEF2F2"},
                {'range': [60, 85], 'color': "#FFFBEB"},
                {'range': [85, 100], 'color': "#ECFDF5"}
            ]
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=15, r=15, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
