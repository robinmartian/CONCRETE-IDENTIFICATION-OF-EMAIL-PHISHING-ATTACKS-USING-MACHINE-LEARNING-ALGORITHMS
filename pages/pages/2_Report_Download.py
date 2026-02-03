import os
import re
import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF

st.set_page_config(page_title="Report Download", layout="wide")

if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
    st.error("Not logged in. Go to the Detector page and enter the password.")
    st.stop()

METRICS_PATH = "metrics.pkl"
if not os.path.exists(METRICS_PATH):
    st.error("metrics.pkl not found. Run train_models.py first.")
    st.stop()

metrics_all = joblib.load(METRICS_PATH)

st.title("Report Download")

def metrics_table(metrics_all):
    rows = []
    for name in ["Naive Bayes", "SVM", "Hybrid MLP->SVM"]:
        m = metrics_all[name]["metrics"]
        rows.append({
            "Model": name,
            "Accuracy (%)": round(m["Accuracy"] * 100, 2),
            "Precision (%)": round(m["Precision"] * 100, 2),
            "Recall (%)": round(m["Recall"] * 100, 2),
            "F1 (%)": round(m["F1"] * 100, 2),
        })
    return pd.DataFrame(rows)

def pdf_safe(text: str) -> str:
    text = text.replace("→", "->")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^\x00-\xFF]", "", text)
    return text

def build_pdf(df_metrics, metrics_all):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Phishing Detection Report (3 Models)", ln=True, align="C")

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    intro = (
        "Models: Naive Bayes, Linear SVM, Hybrid MLP->SVM. "
        "Metrics are based on a held-out test split saved during training."
    )
    pdf.multi_cell(0, 6, pdf_safe(intro))

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary Metrics", ln=True)

    pdf.set_font("Helvetica", "", 10)
    cols = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"]
    widths = [60, 28, 28, 28, 20]

    for c, w in zip(cols, widths):
        pdf.cell(w, 7, pdf_safe(c), border=1)
    pdf.ln()

    for _, r in df_metrics.iterrows():
        pdf.cell(widths[0], 7, pdf_safe(str(r["Model"])), border=1)
        pdf.cell(widths[1], 7, f'{r["Accuracy (%)"]:.2f}', border=1)
        pdf.cell(widths[2], 7, f'{r["Precision (%)"]:.2f}', border=1)
        pdf.cell(widths[3], 7, f'{r["Recall (%)"]:.2f}', border=1)
        pdf.cell(widths[4], 7, f'{r["F1 (%)"]:.2f}', border=1)
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Confusion Matrices (Safe, Phishing)", ln=True)

    for name in ["Naive Bayes", "SVM", "Hybrid MLP->SVM"]:
        cm = metrics_all[name]["cm"]
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, pdf_safe(name), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, pdf_safe(f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"))

    return pdf.output(dest="S").encode("latin1", errors="ignore")

df_metrics = metrics_table(metrics_all)

st.subheader("Metrics table")
st.dataframe(df_metrics, use_container_width=True)

csv_bytes = df_metrics.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download metrics CSV",
    data=csv_bytes,
    file_name="model_metrics_summary.csv",
    mime="text/csv",
    use_container_width=True,
)

st.subheader("PDF report")
pdf_bytes = build_pdf(df_metrics, metrics_all)
st.download_button(
    "Download PDF report",
    data=pdf_bytes,
    file_name="phishing_detection_report.pdf",
    mime="application/pdf",
    use_container_width=True,
)
