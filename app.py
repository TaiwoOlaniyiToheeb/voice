"""
app.py
-------
Voice & Text Command Classifier - Streamlit web app.

Run locally with:
    streamlit run app.py

Deploy for free at: https://share.streamlit.io
"""

import os
import sys
import tempfile

import streamlit as st
import joblib
import plotly.graph_objects as go

# Make src/ importable regardless of where streamlit is launched from
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocess import clean_text  # noqa: E402
from predict import ACTION_DESCRIPTIONS  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "command_classifier.joblib")

# ======================================================================
# THEME CONSTANTS
# Centralized here so colors are consistent everywhere they're referenced
# in both CSS and Python-side logic (e.g. confidence-based coloring).
# ======================================================================
COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#7C3AED"
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"
COLOR_BG = "#072F56"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#6B7280"
GRADIENT_PRIMARY = f"linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%)"

APP_VERSION = "v1.0.0"


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


import numpy as np

def classify_text(model, raw_text):
    cleaned = clean_text(raw_text)

    predicted = model.predict([cleaned])[0]

    # If classifier supports probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]

    # For LinearSVC
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([cleaned])

        # multiclass vs binary
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T

        scores = scores[0]

        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

    else:
        probs = np.ones(len(model.classes_)) / len(model.classes_)

    ranked = sorted(
        zip(model.classes_, probs),
        key=lambda x: x[1],
        reverse=True,
    )

    confidence = dict(ranked)[predicted]

    return predicted, confidence, ranked

def transcribe_uploaded_audio(file_bytes, filename, engine="auto"):
    """
    Saves the uploaded audio to a temp file and transcribes it using the
    same speech recognition logic as voice_predict.py (Google online,
    falling back to offline PocketSphinx automatically).
    """
    import speech_recognition as sr

    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)

        if engine in ("auto", "google"):
            try:
                return recognizer.recognize_google(audio), "google (online)"
            except sr.UnknownValueError:
                return None, None
            except sr.RequestError:
                pass  # fall through to offline engine

        try:
            return recognizer.recognize_sphinx(audio), "pocketsphinx (offline)"
        except (sr.UnknownValueError, OSError):
            return None, None
    finally:
        os.remove(tmp_path)


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Voice & Text Command Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
# CUSTOM CSS
# Every visual element below (cards, buttons, badges, tabs, progress bars,
# footer, upload area) is styled here. 
# ======================================================================
st.markdown(
    f"""
    <style>
    /* ---------- Global background & typography ---------- */
    .stApp {{
        background: {COLOR_BG};
    }}
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: {COLOR_TEXT};
    }}
    #MainMenu, footer {{visibility: hidden;}}

    /* ---------- Hero header ---------- */
    .hero-container {{
        background: {GRADIENT_PRIMARY};
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
        text-align: center;
    }}
    .hero-title {{
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }}
    .hero-subtitle {{
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 640px;
        margin: 0 auto;
    }}
    .ai-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.18);
        color: white;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.35);
    }}

    /* ---------- Feature badges row ---------- */
    .badge-row {{
        display: flex;
        gap: 0.6rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1.1rem;
    }}
    .feature-badge {{
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.25);
    }}

    /* ---------- Generic card ---------- */
    .app-card {{
        background: {COLOR_CARD};
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
        border: 1px solid #EEF2F7;
        margin-bottom: 1.2rem;
    }}
    .app-card h4 {{
        margin-top: 0;
    }}

    /* ---------- Result card ---------- */
    .result-card {{
        background: {COLOR_CARD};
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 6px 20px rgba(17, 24, 39, 0.08);
        border-left: 5px solid {COLOR_SUCCESS};
        margin-top: 1rem;
        animation: fadeIn 0.4s ease-in-out;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .result-intent {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
        margin-bottom: 0.1rem;
    }}
    .result-desc {{
        color: {COLOR_MUTED};
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }}
    .confidence-label {{
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }}

    /* ---------- Buttons ---------- */
    div[data-testid="stButton"] > button {{
        background: {GRADIENT_PRIMARY};
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.6rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div[data-testid="stButton"] > button:hover {{
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
        color: white;
    }}
    div[data-testid="stButton"] > button:active {{
        transform: scale(0.98);
    }}

    /* ---------- Text input ---------- */
    div[data-testid="stTextInput"] input {{
        border-radius: 12px !important;
        border: 1.5px solid #E2E8F0 !important;
        padding: 0.8rem 1rem !important;
        font-size: 1.05rem !important;
        box-shadow: none !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-color: {COLOR_PRIMARY} !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.4rem;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: white;
        border-radius: 12px 12px 0 0;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        color: {COLOR_MUTED};
        border: 1px solid #EEF2F7;
        border-bottom: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: {GRADIENT_PRIMARY};
        color: white !important;
    }}

    /* ---------- Progress bar ---------- */
    div[data-testid="stProgress"] > div > div {{
        background: {GRADIENT_PRIMARY};
        border-radius: 999px;
    }}
    div[data-testid="stProgress"] {{
        border-radius: 999px;
    }}

    /* ---------- File uploader (drag & drop) ---------- */
    div[data-testid="stFileUploader"] {{
        border: 2px dashed #C7D2FE;
        border-radius: 16px;
        padding: 1.2rem;
        background: #F5F7FF;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: {COLOR_PRIMARY};
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: #0F172A;
    }}
    section[data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}
    .sidebar-badge {{
        display: inline-block;
        background: rgba(124, 58, 237, 0.25);
        border: 1px solid rgba(124, 58, 237, 0.5);
        color: #E2E8F0 !important;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin: 0.15rem 0.15rem 0.15rem 0;
    }}

    /* ---------- Footer ---------- */
    .app-footer {{
        text-align: center;
        padding: 1.6rem 0 0.5rem 0;
        color: {COLOR_MUTED};
        font-size: 0.85rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 2rem;
    }}
    .tech-pill {{
        display: inline-block;
        background: #EEF2FF;
        color: {COLOR_PRIMARY};
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# HELPER: confidence -> color
# ======================================================================
def confidence_color(score: float) -> str:
    if score >= 0.7:
        return COLOR_SUCCESS
    elif score >= 0.4:
        return COLOR_WARNING
    return COLOR_DANGER


def render_prediction_card(raw_text, predicted, confidence, ranked):
    """Renders the full prediction result: intent, description, confidence
    bar, and a horizontal Plotly bar chart of the top 5 candidate intents."""
    color = confidence_color(confidence)

    st.markdown(
        f"""
        <div class="result-card" style="border-left-color:{color};">
            <div style="display:flex; align-items:center; gap:0.5rem;">
                <span style="font-size:1.4rem;">✅</span>
                <div>
                    <div class="result-intent">{predicted.replace('_', ' ').title()}</div>
                    <div class="result-desc">{ACTION_DESCRIPTIONS.get(predicted, '')}</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="confidence-label">Confidence</div>', unsafe_allow_html=True)
        st.progress(min(float(confidence), 1.0))
    with col2:
        st.metric(label="Score", value=f"{confidence * 100:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Top-5 Plotly horizontal bar chart ----
    top5 = ranked[:5]
    labels = [intent.replace("_", " ").title() for intent, _ in top5][::-1]
    values = [prob * 100 for _, prob in top5][::-1]
    colors = [confidence_color(p) for _, p in top5][::-1]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont = dict(color = "#000000", size = 13)
            
        )
    
    )
    
    fig.update_layout(
        title="Top 5 Predicted Intents",
        xaxis_title="Confidence (%)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        plot_bgcolor="#F5F5F5",
        paper_bgcolor="#072F56",
        font=dict(color='black', size=13),
        xaxis=dict(range=[0, 100], gridcolor="#000000"),
    )
    st.plotly_chart(fig, width="stretch")


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("### 🎙️ Project Info")
    st.markdown(
        "AI system that classifies short **Nigerian Pidgin / English** "
        "commands into app actions, built for low-literacy users who "
        "prefer voice/text over navigating menus."
    )
    st.markdown("---")

    st.markdown("### 🧠 Model Info")
    st.markdown(
        """
        - **Algorithm:** Support Vector Machine
        - **Features:** TF-IDF (word + bigram)
        - **Intents:** 12
        - **Test accuracy:** ~99% (template set) / 89% (unseen phrasing)
        """
    )
    st.markdown("---")

    st.markdown("### 🌍 Supported Languages")
    st.markdown('<span class="sidebar-badge">English</span><span class="sidebar-badge">Nigerian Pidgin</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🎛️ Input Methods")
    st.markdown('<span class="sidebar-badge">⌨️ Text</span><span class="sidebar-badge">🎤 Voice (upload)</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"**Version:** {APP_VERSION}")
    st.markdown("---")

    st.markdown("### 👤 Author")
    st.markdown("**Taiwo Olaniyi Toheeb**")
    st.markdown(
        """
        🔗 [Portfolio](https://taiwoolaniyitoheeb.github.io/) &nbsp;|&nbsp; [GitHub](https://github.com/TaiwoOlaniyiToheeb) &nbsp;|&nbsp; [LinkedIn](https://www.linkedin.com/in/toheebolaniyitaiwo/)
        📧 [Email](otoheebtaiwo@gmail.com)
        """
    )


# ======================================================================
# HERO HEADER
# ======================================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="ai-badge">✨ AI-POWERED INTENT RECOGNITION</div>
        <div class="hero-title">🎙️ Voice & Text Command Classifier</div>
        <div class="hero-subtitle">
            AI-powered intent recognition for English and Nigerian Pidgin voice/text commands.
        </div>
        <div class="badge-row">
            <span class="feature-badge">Support Vector Machine</span>
            <span class="feature-badge">TF-IDF</span>
            <span class="feature-badge">12 Intents</span>
            <span class="feature-badge">Voice + Text</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model = get_model()


# ======================================================================
# TABS
# ======================================================================
tab_text, tab_audio, tab_about = st.tabs(["⌨️  Text Command", "🎤  Voice Command", "ℹ️  About"])

# ----------------------------------------------------------------
# TAB 1: TEXT COMMAND
# ----------------------------------------------------------------
with tab_text:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### ⌨️ Type a Command")
    st.caption("Try Pidgin or English — e.g. \"abeg check my balance\" or \"send money to my brother\".")

    example_cols = st.columns(3)
    examples = ["abeg check my balance", "send 2k give my mama", "call my broda"]
    for col, ex in zip(example_cols, examples):
        if col.button(ex, key=f"ex_{ex}"):
            st.session_state["text_input"] = ex

    user_text = st.text_input(
        "Command",
        key="text_input",
        placeholder="Type a command such as 'check my balance'...",
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 1])
    classify_clicked = col_a.button("🚀 Classify Command", key="classify_btn")
    clear_clicked = col_b.button("🔄 Clear Input", key="clear_btn")

    if clear_clicked:
        st.session_state["text_input"] = ""
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if (classify_clicked or user_text) and user_text.strip():
        predicted, confidence, ranked = classify_text(model, user_text)
        render_prediction_card(user_text, predicted, confidence, ranked)
    elif classify_clicked and not user_text.strip():
        st.warning("⚠️ Please type a command first.")

# ----------------------------------------------------------------
# TAB 2: VOICE COMMAND
# ----------------------------------------------------------------
with tab_audio:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 🎤 Upload a Voice Command")
    st.caption("Drag and drop, or browse for a short audio clip. Supported formats: **.wav, .flac, .aiff**")

    uploaded = st.file_uploader(
        "Drop your audio file here",
        type=["wav", "flac", "aiff"],
        label_visibility="collapsed",
    )
    analyze_clicked = st.button("🎤 Analyze Audio", key="analyze_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        st.audio(uploaded)

        if analyze_clicked:
            with st.spinner("🔊 Transcribing and classifying your command..."):
                text, engine_used = transcribe_uploaded_audio(uploaded.getvalue(), uploaded.name)

            if text is None:
                st.error("❌ Could not understand the audio. Try a clearer recording or a shorter command.")
            else:
                st.success(f"✅ Transcribed with **{engine_used}**: \"{text}\"")
                predicted, confidence, ranked = classify_text(model, text)
                render_prediction_card(text, predicted, confidence, ranked)
    elif analyze_clicked:
        st.warning("⚠️ Please upload an audio file first.")

    st.info(
        "💡 **Want live microphone input?** Run `python src/voice_predict.py --mic` "
        "locally on your own machine — that path captures directly from your "
        "microphone instead of needing a file upload."
    )

# ----------------------------------------------------------------
# TAB 3: ABOUT
# ----------------------------------------------------------------
with tab_about:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Project Overview")
    st.write(
        "Low-literacy users in Nigeria often find multi-step app menus "
        "difficult to navigate. This project builds the \"brain\" behind a "
        "simpler interaction model: say or type what you want, and the app "
        "figures out what to do."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### ✨ Features")
        st.markdown(
            """
            - Classifies Pidgin **and** English commands
            - Text and voice (audio upload) input
            - Real-time confidence scoring
            - Automatic online → offline speech recognition fallback
            - Interactive CLI tools included in the full project
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Supported Intents (12)")
        intents = [
            "check_balance", "send_money", "buy_airtime", "buy_data",
            "call_contact", "open_app", "check_weather", "set_reminder",
            "play_music", "stop_action", "greeting", "help_request",
        ]
        badges_html = "".join(f'<span class="tech-pill">{i}</span>' for i in intents)
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Model Architecture")
        st.markdown(
            """
            1. **Text cleaning** — lowercase, strip punctuation
            2. **TF-IDF vectorization** — word + bigram features
            3. **Classifier** — SVM (selected after
               comparing against Logistics Regression and Naive Bayes with 5-fold
               cross-validation)
            4. **Prediction** — intent + calibrated confidence score
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Evaluation Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Template test set", "~99%")
        m2.metric("Unseen phrasing set", "~89%")
        st.caption(
            "Both numbers are reported deliberately: the unseen/hand-written "
            "test set is the more honest measure of real-world generalization."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 🚀 Future Improvements")
    st.markdown(
        """
        - Collect real user commands to close the generalization gap further
        - Add a fallback "unclear" intent with a confidence threshold
        - Live in-browser microphone recording (currently: file upload, or
          `--mic` in the local CLI)
        - Fine-tune a small transformer (e.g. DistilBERT) as data grows
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### ☁️ Deployment")
    st.write(
        "Hosted for free on Streamlit Community Cloud, connected directly "
        "to the GitHub repository below."
    )
    st.markdown("[📂 View the full project on GitHub](https://github.com/TaiwoOlaniyiToheeb/voice-text_command_classifier)")
    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================================
# FOOTER
# ======================================================================
st.markdown(
    f"""
    <div class="app-footer">
        Developed by <strong>Taiwo Olaniyi Toheeb</strong><br/>
        Built with
        <span class="tech-pill">Python</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">Scikit-learn</span>
        <span class="tech-pill">SpeechRecognition</span>
        <span class="tech-pill">Joblib</span>
        <span class="tech-pill">TF-IDF</span>
        <span class="tech-pill">SVC</span>
    </div>
    """,
    unsafe_allow_html=True,
)









