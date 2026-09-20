from flask import Flask, render_template, request, jsonify  # added jsonify
from flask_cors import CORS                                  # added CORS
import pickle
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
import difflib
import shap
import os
from dotenv import load_dotenv
import hashlib
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
CORS(app)  # allows Chrome extension to call your Flask server

# ===== LOAD FILES =====
with open("vectorizernew.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("phishing_modelnew.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalernew.pkl", "rb") as f:
    scaler = pickle.load(f)


# ===== GEMINI SETUP =====
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini API connected successfully.")
else:
    gemini_client = None
    print("⚠️  GEMINI_API_KEY not set — Gemini explanations will be disabled.")


# ===== GEMINI RESPONSE CACHE =====
_gemini_cache: dict = {}


# ===== SAFE DOMAINS =====
safe_domains = [
    "google.com", "amazon.com", "amazon.in", "amazon.co.uk",
    "paypal.com", "paypal.co.uk",
    "github.com", "facebook.com", "linkedin.com",
    "microsoft.com", "apple.com",
    "zoom.us", "notion.so", "duckduckgo.com", "canva.com", "figma.com"
]

# ===== FREE HOSTING DOMAINS USED FOR PHISHING =====
SUSPICIOUS_FREE_HOSTS = [
    'my3gb.com', 'freehosting.com', '000webhostapp.com', 'weebly.com',
    'wixsite.com', 'blogspot.com', 'wordpress.com', 'netlify.app',
    'glitch.me', 'replit.dev', 'vercel.app', 'web.app',
    'firebaseapp.com', 'pages.dev', 'surge.sh'
]

# ===== FEATURE NAMES =====
feature_names = [
    "URL Length","Dot Count","Hyphen Count","Digit Count",
    "Contains 'login'","Contains 'secure'","Contains 'verify'",
    "Contains 'account'","Contains 'update'","Excessive Dots (>4)",
    "Contains '@'","Excessive Hyphens (>2)","Too Many Subdomains",
    "Suspicious TLD (.xyz/.tk/.ml/.ga)","Lookalike Domain",
    "Scam Word Count","Multiple Scam Words","Phishing Phrase Count",
    "Contains Phishing Phrase"
]

CONTINUOUS_FEATURES = {
    "URL Length","Dot Count","Hyphen Count",
    "Digit Count","Scam Word Count","Phishing Phrase Count"
}


# ===== LOOKALIKE =====
def is_lookalike(domain):
    brands = ['google', 'amazon', 'paypal', 'facebook', 'microsoft', 'apple']
    for b in brands:
        ratio = difflib.SequenceMatcher(None, domain, b).ratio()
        if 0.70 < ratio < 1.0:
            return 1
    return 0


# ===== FREE HOST CHECK =====
def is_free_host(domain):
    parts = domain.split('.')
    if len(parts) >= 2:
        root = '.'.join(parts[-2:])
        if root in SUSPICIOUS_FREE_HOSTS:
            return True
    return False


# ===== FEATURE EXTRACTION =====
def extract_features(url):
    url = url.lower()
    domain = url.split('/')[0]
    parts = domain.split('.')
    base_domain = parts[0]

    scam_words = ['free','money','win','prize','gift','bonus','offer','giveaway']
    scam_count = sum(word in url for word in scam_words)

    phrase_patterns = [
        'account-verification','verification-required','user-verification',
        'limited-offer','offer-free','free-subscription',
        'confirm-account','update-details','login-support','security-alert'
    ]
    phrase_count = sum(p in url for p in phrase_patterns)

    return [
        len(url),url.count('.'),url.count('-'),
        sum(c.isdigit() for c in url),
        int('login' in url),int('secure' in url),
        int('verify' in url),int('account' in url),
        int('update' in url),int(url.count('.') > 4),
        int('@' in url),int(url.count('-') > 2),
        int(len(parts) > 4),
        int(domain.endswith(('.xyz','.tk','.ml','.ga'))),
        is_lookalike(base_domain),
        scam_count,int(scam_count >= 2),
        phrase_count,int(phrase_count >= 1)
    ]


# ===== SHAP EXPLAINER SETUP =====
n_tfidf_features = len(vectorizer.get_feature_names_out())
n_handcrafted = len(feature_names)
n_total = n_tfidf_features + n_handcrafted

background = np.zeros((1, n_total))
explainer = shap.LinearExplainer(model, background)


# ===== SHAP EXPLANATION =====
def get_shap_explanation(combined_input, url_features_raw, prediction):
    if hasattr(combined_input, "toarray"):
        combined_dense = combined_input.toarray()
    else:
        combined_dense = np.array(combined_input)

    shap_values = explainer.shap_values(combined_dense)
    handcrafted_shap = shap_values[0, -n_handcrafted:]
    show_direction = "phishing" if prediction == "bad" else "safe"

    explanations = []
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


def make_item(name, shap_val, raw_val, direction):
    return {
        "name": name,
        "shap": round(float(shap_val), 3),
        "value": round(float(raw_val), 2),
        "direction": direction
    }


# ===== GEMINI EXPLANATION =====
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
                f"{s['name']} ({'Risk' if s['direction']=='phishing' else 'Safe'})"
                for s in top_signals
            )
        else:
            signals_text = "free hosting" if free_host else "URL patterns"

        prompt = (
            f'URL: {url} | Result: {verdict_text} | Flags: {signals_text}\n'
            f'Explain in 2 sentences why it is {verdict_text}.'
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=80
            )
        )

        result = response.text.strip()
        _gemini_cache[cache_key] = result
        return result

    except Exception:
        return None


# ===== PREDICT =====
def smart_predict(url):
    cleaned = re.sub(r'^https?://(www\.)?', '', url.lower())
    domain = cleaned.split('/')[0]

    for safe in safe_domains:
        if domain == safe or domain.endswith('.' + safe):
            gemini_text = get_gemini_explanation(url, "good", [])
            return "good", [], gemini_text

    if is_free_host(domain):
        gemini_text = get_gemini_explanation(url, "bad", [], True)
        return "bad", [], gemini_text

    text_vec = vectorizer.transform([cleaned])
    url_feat_raw = np.array([extract_features(cleaned)])
    url_feat_scaled = scaler.transform(url_feat_raw)
    combined = hstack([text_vec, url_feat_scaled])

    prediction = model.predict(combined)[0]
    shap_signals = get_shap_explanation(combined, url_feat_raw, prediction)
    gemini_text = get_gemini_explanation(url, prediction, shap_signals)

    return prediction, shap_signals, gemini_text


# ===== MAIN ROUTE (web app) =====
@app.route("/", methods=["GET","POST"])
def index():
    result = None
    explanation = []
    gemini_text = None
    url_input = ""

    if request.method == "POST":
        url = request.form["url"].strip()
        url_input = url

        prediction, explanation, gemini_text = smart_predict(url)

        if prediction == "bad":
            result = "⚠️ Phishing Website Detected"
        else:
            result = "✅ Safe Website"

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        gemini_text=gemini_text,
        url_input=url_input
    )


# ===== API ROUTE (Chrome extension) =====
@app.route("/api/check", methods=["POST"])
def api_check():
    """
    JSON endpoint for the Chrome extension.
    Receives: { "url": "https://example.com" }
    Returns:  { "prediction": "good"/"bad", "explanation": [...], "gemini_text": "..." }
    """
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data["url"].strip()

    try:
        prediction, explanation, gemini_text = smart_predict(url)
        return jsonify({
            "prediction": prediction,
            "explanation": explanation,
            "gemini_text": gemini_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
