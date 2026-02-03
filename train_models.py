# train_models.py
# Trains and saves:
# - TF-IDF vectorizer
# - Naive Bayes
# - Linear SVM
# - Hybrid MLP -> SVM
# Also saves metrics.pkl for the Streamlit app (no training inside Streamlit)

import os
import re
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# =========================
# 1) SETTINGS
# =========================
DATA_PATH = "Phishing_Email.csv"

VECTORIZER_OUT = "vectorizer.pkl"
NB_OUT = "nb_model.pkl"
SVM_OUT = "svm_model.pkl"
MLP_OUT = "mlp_model.pkl"
HYBRID_OUT = "hybrid_model.pkl"
METRICS_OUT = "metrics.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF settings (tune these if you want)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Hybrid MLP settings (tune these if you want)
MLP_HIDDEN = (50,)
MLP_MAX_ITER = 150

# Labels order used for confusion matrix
LABELS_ORDER = ["Safe Email", "Phishing Email"]
POS_LABEL = "Phishing Email"


# =========================
# 2) TEXT CLEANING
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # urls
    text = re.sub(r"\S+@\S+", " ", text)                # emails
    text = re.sub(r"<.*?>", " ", text)                  # html tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # non alphanum
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# 3) LOAD DATASET
# =========================
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in this folder")

    df = pd.read_csv(path)

    required = {"Email Text", "Email Type"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Email Text' and 'Email Type'")

    df = df.copy()
    df["Cleaned"] = df["Email Text"].apply(clean_text)

    return df


# =========================
# 4) TRAIN + EVAL HELPERS
# =========================
def evaluate(model, X_test, y_test, *, use_hybrid_input=False, mlp_model=None):
    """
    If use_hybrid_input is True, model expects MLP probabilities as input.
    """
    if use_hybrid_input:
        if mlp_model is None:
            raise ValueError("mlp_model must be provided for hybrid evaluation.")
        X_in = mlp_model.predict_proba(X_test)
    else:
        X_in = X_test

    y_pred = model.predict(X_in)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, pos_label=POS_LABEL)),
        "Recall": float(recall_score(y_test, y_pred, pos_label=POS_LABEL)),
        "F1": float(f1_score(y_test, y_pred, pos_label=POS_LABEL)),
    }

    cm = confusion_matrix(y_test, y_pred, labels=LABELS_ORDER).tolist()

    report = classification_report(
        y_test, y_pred, target_names=LABELS_ORDER, digits=4
    )

    return metrics, cm, report


def print_dataset_stats(df: pd.DataFrame):
    total = len(df)
    phishing = int((df["Email Type"] == "Phishing Email").sum())
    safe = int((df["Email Type"] == "Safe Email").sum())
    print(f"Rows: {total}  -  Phishing: {phishing}  -  Safe: {safe}")


# =========================
# 5) MAIN
# =========================
def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print_dataset_stats(df)

    X = df["Cleaned"]
    y = df["Email Type"]

    print("Splitting train/test...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # -------------------------
    # Train Models
    # -------------------------
    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    print("Training Linear SVM...")
    svm_model = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train, y_train)

    print("Training MLP (for hybrid features)...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    mlp_model.fit(X_train, y_train)

    print("Training Hybrid SVM on MLP probabilities...")
    X_train_hybrid = mlp_model.predict_proba(X_train)
    hybrid_model = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    hybrid_model.fit(X_train_hybrid, y_train)

    # -------------------------
    # Evaluate Models
    # -------------------------
    print("Evaluating models...")
    nb_metrics, nb_cm, nb_report = evaluate(nb_model, X_test, y_test)
    svm_metrics, svm_cm, svm_report = evaluate(svm_model, X_test, y_test)
    hy_metrics, hy_cm, hy_report = evaluate(
        hybrid_model, X_test, y_test, use_hybrid_input=True, mlp_model=mlp_model
    )

    metrics_all = {
        "Naive Bayes": {
            "metrics": nb_metrics,
            "cm": nb_cm,
            "report": nb_report,
            "labels_order": LABELS_ORDER,
        },
        "SVM": {
            "metrics": svm_metrics,
            "cm": svm_cm,
            "report": svm_report,
            "labels_order": LABELS_ORDER,
        },
        "Hybrid MLP->SVM": {   # ASCII arrow to avoid unicode issues later
            "metrics": hy_metrics,
            "cm": hy_cm,
            "report": hy_report,
            "labels_order": LABELS_ORDER,
        },
    }

    # -------------------------
    # Save Artifacts
    # -------------------------
    print("Saving models and vectorizer...")
    joblib.dump(vectorizer, VECTORIZER_OUT)
    joblib.dump(nb_model, NB_OUT)
    joblib.dump(svm_model, SVM_OUT)
    joblib.dump(mlp_model, MLP_OUT)
    joblib.dump(hybrid_model, HYBRID_OUT)

    print("Saving metrics...")
    joblib.dump(metrics_all, METRICS_OUT)

    print("Done. Files created:")
    print(f"- {VECTORIZER_OUT}")
    print(f"- {NB_OUT}")
    print(f"- {SVM_OUT}")
    print(f"- {MLP_OUT}")
    print(f"- {HYBRID_OUT}")
    print(f"- {METRICS_OUT}")


if __name__ == "__main__":
    main()
