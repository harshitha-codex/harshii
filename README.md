# LinkSus

LinkSus is a phishing URL detection app built with Python, scikit-learn, SHAP, and Streamlit.

## What this project does

- Detects phishing URLs from user input
- Uses handcrafted URL features and TF-IDF text features
- Shows SHAP-based explanations for the final prediction
- Can be deployed on Streamlit Cloud from GitHub

## Local run

```bash
cd "c:\Users\Administrator\Downloads\core-20260920T171223Z-1-001\core"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Deploy on GitHub + Streamlit Cloud

1. Push this project to a GitHub repository.
2. Go to https://share.streamlit.io
3. Sign in with GitHub.
4. Click "New app".
5. Select the repository, branch, and set the main file to `app.py`.
6. Click "Deploy".

Your app will be available as a live Streamlit URL from the Streamlit Cloud dashboard.

## Project files

- `app.py` — Streamlit app
- `phishing_modelnew.pkl` — trained model
- `vectorizernew.pkl` — TF-IDF vectorizer
- `scalernew.pkl` — feature scaler
- `dataset/phishing_site_urls.csv` — sample data

## Notes

- This project is designed for Streamlit hosting, not GitHub Pages.
- If you want Gemini explanations, set `GEMINI_API_KEY` in your environment or Streamlit secrets.
