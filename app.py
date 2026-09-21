import hashlib
import os
import pickle
import re
from typing import Any

import difflib

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from scipy.sparse import hstack
import shap

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency for Gemini
    genai = None
    types = None

load_dotenv()


@st.cache_resource
def load_model_assets():
    with open("vectorizernew.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("phishing_modelnew.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scalernew.pkl", "rb") as f:
        scaler = pickle.load(f)

    return vectorizer, model, scaler


vectorizer, model, scaler = load_model_assets()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and genai is not None and types is not None:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

_gemini_cache: dict[str, str] = {}

safe_domains = [
    "google.com", "amazon.com", "amazon.in", "amazon.co.uk",
    "paypal.com", "paypal.co.uk",
    "github.com", "facebook.com", "linkedin.com",
    "microsoft.com", "apple.com",
    "zoom.us", "notion.so", "duckduckgo.com", "canva.com", "figma.com",
]

SUSPICIOUS_FREE_HOSTS = [
    "my3gb.com", "freehosting.com", "000webhostapp.com", "weebly.com",
    "wixsite.com", "blogspot.com", "wordpress.com", "netlify.app",
    "glitch.me", "replit.dev", "vercel.app", "web.app",
    "firebaseapp.com", "pages.dev", "surge.sh",
]

feature_names = [
    "URL Length", "Dot Count", "Hyphen Count", "Digit Count",
    "Contains 'login'", "Contains 'secure'", "Contains 'verify'",
    "Contains 'account'", "Contains 'update'", "Excessive Dots (>4)",
    "Contains '@'", "Excessive Hyphens (>2)", "Too Many Subdomains",
    "Suspicious TLD (.xyz/.tk/.ml/.ga)", "Lookalike Domain",
    "Scam Word Count", "Multiple Scam Words", "Phishing Phrase Count",
    "Contains Phishing Phrase",
]

CONTINUOUS_FEATURES = {
    "URL Length", "Dot Count", "Hyphen Count",
    "Digit Count", "Scam Word Count", "Phishing Phrase Count",
}


def is_lookalike(domain):
    brands = ["google", "amazon", "paypal", "facebook", "microsoft", "apple"]
    for brand in brands:
        ratio = difflib.SequenceMatcher(None, domain, brand).ratio()
        if 0.70 < ratio < 1.0:
            return 1
    return 0


def is_free_host(domain):
    parts = domain.split(".")
    if len(parts) >= 2:
        root = ".".join(parts[-2:])
        if root in SUSPICIOUS_FREE_HOSTS:
            return True
    return False


def extract_features(url):
    url = url.lower()
    domain = url.split("/")[0]
    parts = domain.split(".")
    base_domain = parts[0]

    scam_words = ["free", "money", "win", "prize", "gift", "bonus", "offer", "giveaway"]
    scam_count = sum(word in url for word in scam_words)

    phrase_patterns = [
        "account-verification", "verification-required", "user-verification",
        "limited-offer", "offer-free", "free-subscription",
        "confirm-account", "update-details", "login-support", "security-alert",
    ]
    phrase_count = sum(p in url for p in phrase_patterns)

    return [
        len(url), url.count("."), url.count("-"),
        sum(c.isdigit() for c in url),
        int("login" in url), int("secure" in url),
        int("verify" in url), int("account" in url),
        int("update" in url), int(url.count(".") > 4),
        int("@" in url), int(url.count("-") > 2),
        int(len(parts) > 4),
        int(domain.endswith((".xyz", ".tk", ".ml", ".ga"))),
        is_lookalike(base_domain),
        scam_count, int(scam_count >= 2),
        phrase_count, int(phrase_count >= 1),
    ]


@st.cache_data
def get_shap_explainer():
    n_tfidf_features = len(vectorizer.get_feature_names_out())
    n_handcrafted = len(feature_names)
    n_total = n_tfidf_features + n_handcrafted
    background = np.zeros((1, n_total))
    return shap.LinearExplainer(model, background)


explainer = get_shap_explainer()


def make_item(name, shap_val, raw_val, direction):
    return {
        "name": name,
        "shap": round(float(shap_val), 3),
        "value": round(float(raw_val), 2),
        "direction": direction,
    }


def get_shap_explanation(combined_input, url_features_raw, prediction):
    if hasattr(combined_input, "toarray"):
        combined_dense = combined_input.toarray()
    else:
        combined_dense = np.array(combined_input)

    shap_values = explainer.shap_values(combined_dense)
    handcrafted_shap = shap_values[0, -len(feature_names):]
    show_direction = "phishing" if prediction == "bad" else "safe"

    explanations: list[dict[str, Any]] = []
    for name, shap_val, raw_val in zip(feature_names, handcrafted_shap, url_features_raw[0]):
        if abs(shap_val) <= 0.01:
            continue

        is_continuous = name in CONTINUOUS_FEATURES
        if shap_val > 0:
            if show_direction != "phishing":
                continue
            if is_continuous and raw_val > 0:
                explanations.append(make_item(name, shap_val, raw_val, "phishing"))
            elif not is_continuous and raw_val == 1:
                explanations.append(make_item(name, shap_val, raw_val, "phishing"))
        else:
            if show_direction != "safe":
                continue
            if is_continuous:
                explanations.append(make_item(name, shap_val, raw_val, "safe"))
            elif raw_val == 0:
                explanations.append(make_item(name, shap_val, raw_val, "safe"))

    explanations.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return explanations[:5]


def get_gemini_explanation(url, prediction, shap_signals, free_host=False):
    if not gemini_client:
        return None

    verdict_text = "PHISHING" if prediction == "bad" else "SAFE"
    cache_key = hashlib.md5(f"{url.lower()}:{verdict_text}".encode()).hexdigest()

    if cache_key in _gemini_cache:
        return _gemini_cache[cache_key]

    try:
        top_signals = shap_signals[:2]
        if top_signals:
            signals_text = ", ".join(
                f"{signal['name']} ({'Risk' if signal['direction'] == 'phishing' else 'Safe'})"
                for signal in top_signals
            )
        else:
            signals_text = "free hosting" if free_host else "URL patterns"

        prompt = (
            f"URL: {url} | Result: {verdict_text} | Flags: {signals_text}\n"
            f"Explain in 2 sentences why it is {verdict_text}."
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=80,
            ),
        )

        result = response.text.strip()
        _gemini_cache[cache_key] = result
        return result
    except Exception:
        return None


def smart_predict(url, return_source=False):
    cleaned = re.sub(r"^https?://(www\.)?", "", url.lower())
    domain = cleaned.split("/")[0]

    for safe in safe_domains:
        if domain == safe or domain.endswith("." + safe):
            res = ("good", [], get_gemini_explanation(url, "good", []))
            return res + ("Safe Domain Whitelist",) if return_source else res

    if is_free_host(domain):
        res = ("bad", [], get_gemini_explanation(url, "bad", [], True))
        return res + ("Suspicious Free-Host Heuristic",) if return_source else res

    text_vec = vectorizer.transform([cleaned])
    url_feat_raw = np.array([extract_features(cleaned)])
    url_feat_scaled = scaler.transform(url_feat_raw)
    combined = hstack([text_vec, url_feat_scaled])

    prediction = model.predict(combined)[0]
    shap_signals = get_shap_explanation(combined, url_feat_raw, prediction)
    gemini_text = get_gemini_explanation(url, prediction, shap_signals)

    res = (prediction, shap_signals, gemini_text)
    return res + ("LinearSVC ML Model + SHAP",) if return_source else res


st.set_page_config(
    page_title="LinkSus — Cyber Defense Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean White Background & Modern Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Clean white background & modern typography */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        background-image: none !important;
    }

    [data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    /* Headings & Text */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #0f172a;
    }

    .brand-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-bottom: 25px;
        padding-top: 5px;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border: 1px solid rgba(37,99,235,0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 8px 20px rgba(37,99,235,0.25);
        margin-bottom: 12px;
    }

    .brand-title {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #0f172a 30%, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .brand-tagline {
        font-size: 11px;
        font-weight: 700;
        color: #64748b !important;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* Verdict Cards */
    .verdict-card {
        padding: 22px 24px;
        border-radius: 14px;
        margin: 18px 0;
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .verdict-safe {
        background: #f0fdf4 !important;
        border: 1.5px solid #86efac !important;
        box-shadow: 0 6px 18px rgba(22,163,74,0.08);
    }

    .verdict-bad {
        background: #fef2f2 !important;
        border: 1.5px solid #fca5a5 !important;
        box-shadow: 0 6px 18px rgba(220,38,38,0.08);
    }

    .verdict-heading {
        font-size: 20px;
        font-weight: 800;
        margin: 0;
    }

    .verdict-safe .verdict-heading { color: #15803d !important; }
    .verdict-bad .verdict-heading { color: #b91c1c !important; }

    .verdict-safe .verdict-sub {
        font-size: 13px;
        color: #166534 !important;
        margin-top: 4px;
    }

    .verdict-bad .verdict-sub {
        font-size: 13px;
        color: #991b1b !important;
        margin-top: 4px;
    }

    /* Feature Pills */
    .feature-pill-risk {
        background: #fee2e2 !important;
        color: #dc2626 !important;
        border: 1px solid #fca5a5 !important;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }

    .feature-pill-safe {
        background: #dcfce7 !important;
        color: #15803d !important;
        border: 1px solid #86efac !important;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }

    /* Gemini AI Container */
    .gemini-container {
        background: #f8fafc !important;
        border: 1.5px solid #c7d2fe !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        box-shadow: 0 4px 14px rgba(99,102,241,0.06);
    }

    .gemini-pill {
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: #e0e7ff !important;
        color: #4338ca !important;
        border: 1px solid #c7d2fe !important;
        padding: 2px 8px;
        border-radius: 12px;
        margin-bottom: 8px;
    }

    .gemini-body {
        color: #334155 !important;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Input styling */
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }

    input[type="text"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #0f172a !important;
        background-color: #ffffff !important;
    }

    input[type="text"]::placeholder {
        color: #94a3b8 !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
    }

    button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
    }

    button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    button[kind="secondary"]:hover {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }

    /* Metrics Cards */
    [data-testid="stMetric"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }

    code {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ LinkSus Security Engine")
    st.caption("Multilayer Phishing Detection & Explainability")
    st.markdown("---")
    st.markdown("**Active Pipeline:**")
    st.markdown("- 🔍 **19 Lexical Heuristics** (Domain, Length, TLDs, Brand lookalikes)")
    st.markdown("- 📊 **TF-IDF Vectorizer** (N-gram sub-patterns)")
    st.markdown("- ⚡ **LinearSVC Model** (Hyperplane boundary)")
    st.markdown("- 💡 **SHAP Explainability** (Linear feature attribution)")
    st.markdown("- 🤖 **Gemini 2.5 AI** (Natural language synthesis)")
    st.markdown("---")
    st.markdown("**Chrome Extension:**")
    st.info("Manifest V3 extension located in `extension/`. Connects to LinkSus backend for instant tab inspection.")

# Main layout
st.markdown(
    """
    <div class="brand-container">
        <div class="brand-icon">🛡️</div>
        <h1 class="brand-title">LinkSus</h1>
        <div class="brand-tagline">Cyber Defense Intelligence // Phishing URL Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Manage state for sample clicks
if "input_url" not in st.session_state:
    st.session_state.input_url = ""
if "trigger_scan" not in st.session_state:
    st.session_state.trigger_scan = False

st.markdown("##### Quick Test Samples")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    if st.button("🌐 Google (Legitimate)", use_container_width=True):
        st.session_state.input_url = "https://google.com"
        st.session_state.trigger_scan = True
with col_s2:
    if st.button("💳 PayPal (Legitimate)", use_container_width=True):
        st.session_state.input_url = "https://paypal.com"
        st.session_state.trigger_scan = True
with col_s3:
    if st.button("⚠️ Phishing Scam URL", use_container_width=True):
        st.session_state.input_url = "http://secure-login-verify-account-update.xyz/login"
        st.session_state.trigger_scan = True
with col_s4:
    if st.button("🚨 Free-Host Abuse", use_container_width=True):
        st.session_state.input_url = "https://account-security-alert.vercel.app"
        st.session_state.trigger_scan = True

# Main input form
col_in, col_btn = st.columns([5, 1])
with col_in:
    url_val = st.text_input(
        "Enter a URL to scan",
        value=st.session_state.input_url,
        placeholder="https://example.com/login",
        label_visibility="collapsed",
    )
with col_btn:
    scan_btn = st.button("Inspect URL", type="primary", use_container_width=True)

should_run = scan_btn or st.session_state.trigger_scan
st.session_state.trigger_scan = False

if should_run:
    target_url = url_val.strip()
    if not target_url:
        st.warning("Please enter or select a URL to inspect.")
    else:
        with st.spinner("Analyzing URL across heuristics, vectorizer, and ML models..."):
            prediction, explanation, gemini_text, source = smart_predict(target_url, return_source=True)

        if prediction == "bad":
            st.markdown(
                f"""
                <div class="verdict-card verdict-bad">
                    <div style="font-size:36px; line-height:1;">⚠️</div>
                    <div>
                        <div class="verdict-heading">PHISHING THREAT DETECTED</div>
                        <div class="verdict-sub">Flagged by: <strong>{source}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-card verdict-safe">
                    <div style="font-size:36px; line-height:1;">🛡️</div>
                    <div>
                        <div class="verdict-heading">LEGITIMATE WEBSITE VERIFIED</div>
                        <div class="verdict-sub">Verified by: <strong>{source}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Overview Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Verdict", "PHISHING" if prediction == "bad" else "SAFE")
        with m2:
            st.metric("Detection Route", source.split()[0])
        with m3:
            st.metric("Evaluated Heuristics", f"{len(feature_names)} features")

        # Explanations
        if explanation:
            st.markdown("#### 🔬 Key Feature Attribution (SHAP Signals)")
            for item in explanation:
                is_risk = item["direction"] == "phishing"
                pill_class = "feature-pill-risk" if is_risk else "feature-pill-safe"
                label_text = "⚠ Risk Factor" if is_risk else "✓ Safe Indicator"

                c_name, c_val, c_shap, c_badge = st.columns([3, 2, 2, 2])
                with c_name:
                    st.markdown(f"**{item['name']}**")
                with c_val:
                    st.markdown(f"`Value: {item['value']}`")
                with c_shap:
                    st.markdown(f"`SHAP: {item['shap']:+.3f}`")
                with c_badge:
                    st.markdown(f'<span class="{pill_class}">{label_text}</span>', unsafe_allow_html=True)

        # Gemini AI Explanation
        if gemini_text:
            st.markdown(
                f"""
                <div class="gemini-container">
                    <span class="gemini-pill">✨ Gemini AI Security Assessment</span>
                    <div class="gemini-body">{gemini_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not gemini_client:
            st.caption("💡 Set `GEMINI_API_KEY` in your `.env` file to enable automated Gemini AI security insights.")

st.markdown("---")
st.caption("LinkSus — Explainable AI Phishing URL Detection // LinearSVC + SHAP + Google Gemini")

