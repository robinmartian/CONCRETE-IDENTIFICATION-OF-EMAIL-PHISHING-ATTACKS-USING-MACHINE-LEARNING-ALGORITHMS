import os
import re
import warnings
from io import StringIO

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time

MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT_SECONDS = 600

warnings.filterwarnings("ignore")

# =========================
# 1) PASSWORD GATE (HARDENED)
# =========================
def password_gate():
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", "").strip()

    if not APP_PASSWORD:
        st.error("APP_PASSWORD is not set")
        st.stop()

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0

    if "last_activity" not in st.session_state:
        st.session_state.last_activity = time.time()

    if st.session_state.auth_ok:
        if time.time() - st.session_state.last_activity > SESSION_TIMEOUT_SECONDS:
            st.session_state.auth_ok = False
            st.session_state.login_attempts = 0
            st.warning("Session expired. Please log in again.")
            st.stop()
        else:
            st.session_state.last_activity = time.time()
            return

    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.error("Too many failed login attempts. Session locked.")
        st.stop()

    with st.sidebar:
        st.header("Access Control")
        pw = st.text_input("Enter password", type="password")

        if pw:
            if pw == APP_PASSWORD:
                st.session_state.auth_ok = True
                st.session_state.login_attempts = 0
                st.session_state.last_activity = time.time()
                st.success("Access granted")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.warning(f"Incorrect password. Attempts remaining: {remaining}")
                st.stop()
        else:
            st.info("Enter password to access the app")
            st.stop()


password_gate()

# =========================
# 2) PATHS
# =========================
DATA_PATH = "Phishing_Email.csv"
VECTORIZER_PATH = "vectorizer.pkl"
NB_MODEL_PATH = "nb_model.pkl"
SVM_MODEL_PATH = "svm_model.pkl"
MLP_MODEL_PATH = "mlp_model.pkl"
HYBRID_MODEL_PATH = "hybrid_model.pkl"
METRICS_PATH = "metrics.pkl"

HYBRID_KEY = "Hybrid MLP->SVM"

# =========================
# 3) HELPERS
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_metrics_key_compat(metrics_all):
    if HYBRID_KEY not in metrics_all and "Hybrid MLP→SVM" in metrics_all:
        metrics_all[HYBRID_KEY] = metrics_all["Hybrid MLP→SVM"]
    return metrics_all


# =========================
# 4) LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    paths = [
        VECTORIZER_PATH,
        NB_MODEL_PATH,
        SVM_MODEL_PATH,
        MLP_MODEL_PATH,
        HYBRID_MODEL_PATH,
        METRICS_PATH,
    ]

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        return None, None, None, None, None, None, missing

    vectorizer = joblib.load(VECTORIZER_PATH)
    nb_model = joblib.load(NB_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    mlp_model = joblib.load(MLP_MODEL_PATH)
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)
    metrics_all = joblib.load(METRICS_PATH)

    metrics_all = ensure_metrics_key_compat(metrics_all)

    return vectorizer, nb_model, svm_model, mlp_model, hybrid_model, metrics_all, []


vectorizer, nb_model, svm_model, mlp_model, hybrid_model, metrics_all, missing = load_artifacts()

if missing:
    st.error("Missing required files:")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

# =========================
# 5) SIDEBAR
# =========================
with st.sidebar:
    st.header("About")
    st.write(
        "This prototype compares three machine learning models on the same phishing email dataset:"
    )
    st.markdown(
        "- Naive Bayes\n"
        "- Linear SVM\n"
        f"- {HYBRID_KEY} (two-stage model)"
    )
    st.info("This is a research prototype focused on model comparison and evaluation.")

    if st.button("Log out"):
        st.session_state.auth_ok = False
        st.rerun()

# =========================
# 6) MAIN TITLE
# =========================
st.title("Phishing Detection and Security Analysis")
st.caption("Models are trained offline. This app loads saved models for prediction.")
st.divider()

# =========================
# DATASET PREVIEW
# =========================
with st.expander("Dataset preview", expanded=False):
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if {"Email Text", "Email Type"}.issubset(df.columns):
            st.dataframe(
                df[["Email Text", "Email Type"]].head(10),
                use_container_width=True
            )
            st.write(
                f"Rows: {len(df)} | "
                f"Phishing: {(df['Email Type']=='Phishing Email').sum()} | "
                f"Safe: {(df['Email Type']=='Safe Email').sum()}"
            )
        else:
            st.warning("CSV missing required columns")
    else:
        st.warning("Phishing_Email.csv not found")

st.divider()

# =========================
# 7) SINGLE EMAIL DETECTION
# =========================
st.subheader("Single Email Detection")

with st.expander("Test a single email", expanded=True):
    col_left, col_right = st.columns([2, 1])

    if st.session_state.get("clear_email"):
        st.session_state.pop("email_input", None)
        del st.session_state["clear_email"]

    with col_left:
        email_text = st.text_area(
            "Paste email content here",
            height=180,
            key="email_input"
        )

    with col_right:
        model_choice = st.selectbox("Select model", ["Naive Bayes", "SVM", HYBRID_KEY])
        predict_clicked = st.button("Predict", use_container_width=True)

    if predict_clicked and email_text.strip():
        clean = clean_text(email_text)
        vec = vectorizer.transform([clean])

        if model_choice == "Naive Bayes":
            model = nb_model
            X_input = vec
        elif model_choice == "SVM":
            model = svm_model
            X_input = vec
        else:
            model = hybrid_model
            X_input = mlp_model.predict_proba(vec)

        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0].max() * 100

        st.session_state["last_prediction"] = {
            "label": pred,
            "confidence": prob
        }
        st.session_state["last_email"] = email_text
        st.session_state["clear_email"] = True
        st.rerun()

    if st.session_state.get("last_prediction"):
        result = st.session_state["last_prediction"]
        if result["label"] == "Phishing Email":
            st.error(
                f"Prediction: {result['label']} ({result['confidence']:.1f}% confidence)"
            )
        else:
            st.success(
                f"Prediction: {result['label']} ({result['confidence']:.1f}% confidence)"
            )

st.caption(
    "Privacy notice: Email content is processed in memory only and is not stored or logged."
)

# =========================
# EXTRA SECURITY ANALYSIS
# =========================
if st.session_state.get("last_email"):
    email_for_analysis = st.session_state["last_email"]

    st.subheader("Security analysis")

    url_count = len(re.findall(r"https?://", email_for_analysis.lower()))

    suspicious_words = [
        "verify", "urgent", "account", "suspend",
        "click", "login", "update", "confirm", "password"
    ]

    suspicious_hits = [
        w for w in suspicious_words if w in email_for_analysis.lower()
    ]

    special_char_count = sum(
        1 for c in email_for_analysis if not c.isalnum() and not c.isspace()
    )

    analysis_df = pd.DataFrame({
        "Indicator": [
            "Number of URLs",
            "Suspicious keywords detected",
            "Detected keywords",
            "Special character count"
        ],
        "Observation": [
            url_count,
            len(suspicious_hits),
            ", ".join(suspicious_hits) if suspicious_hits else "None",
            special_char_count
        ]
    })

    st.table(analysis_df)

    st.caption(
        "This section presents heuristic security indicators commonly "
        "associated with phishing emails. It is shown to complement "
        "machine learning predictions and improve interpretability."
    )

st.divider()

# =========================
# 8) MODEL EVALUATION
# =========================
st.subheader("Model evaluation")


def show_model_results(model_name):
    data = metrics_all[model_name]
    metrics = data["metrics"]
    cm = data["cm"]
    report_text = data["report"]

    st.write("**Metrics**")
    st.write(f"Accuracy: {metrics['Accuracy'] * 100:.2f}%")
    st.write(f"Precision: {metrics['Precision'] * 100:.2f}%")
    st.write(f"Recall: {metrics['Recall'] * 100:.2f}%")
    st.write(f"F1 score: {metrics['F1'] * 100:.2f}%")

    st.write("**Confusion matrix**")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Safe", "Actual Phishing"],
        columns=["Pred Safe", "Pred Phishing"],
    )
    st.table(cm_df)

    st.write("**Classification report**")
    report_df = pd.read_fwf(StringIO(report_text))
    report_df = report_df.dropna().reset_index(drop=True)

    report_df = report_df[
        ~report_df.iloc[:, 0].str.contains("macro avg|weighted avg", na=False)
    ]

    st.dataframe(report_df, use_container_width=True)


with st.expander("Naive Bayes results"):
    show_model_results("Naive Bayes")

with st.expander("SVM results"):
    show_model_results("SVM")

with st.expander(f"{HYBRID_KEY} results"):
    show_model_results(HYBRID_KEY)
