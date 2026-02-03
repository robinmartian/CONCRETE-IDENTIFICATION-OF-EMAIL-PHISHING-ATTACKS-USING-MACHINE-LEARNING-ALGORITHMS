import os
import re
import tempfile
import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF

# Optional matplotlib (PDF only)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# =========================
# PAGE CONFIG & AUTH
# =========================
st.set_page_config(page_title="Results and Charts", layout="wide")

if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
    st.error("Not logged in. Go to the Detector page and enter the password.")
    st.stop()


# =========================
# LOAD METRICS
# =========================
METRICS_PATH = "metrics.pkl"
if not os.path.exists(METRICS_PATH):
    st.error("metrics.pkl not found. Run train_models.py first.")
    st.stop()

metrics_all = joblib.load(METRICS_PATH)

MODEL_ORDER = ["Naive Bayes", "SVM", "Hybrid MLP->SVM"]


# =========================
# HELPERS
# =========================
def metrics_table(metrics_dict):
    rows = []
    for name in MODEL_ORDER:
        if name not in metrics_dict:
            continue
        m = metrics_dict[name]["metrics"]
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


def plot_cm_png(cm, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=150)
    ax.imshow(cm)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Safe", "Phishing"])
    ax.set_yticklabels(["Safe", "Phishing"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=10)

    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name)
    plt.close(fig)
    return tmp.name


def plot_metrics_clustered_png(df_metrics):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    df_plot = df_metrics.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"]]
    df_plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Model")
    ax.set_title("Model Performance Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name)
    plt.close(fig)
    return tmp.name


# =========================
# PDF BUILDER
# =========================
def build_pdf_report(df_metrics, metrics_dict):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    tmp_imgs = []

    # PAGE 1 – INTRO + SUMMARY
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Phishing Email Classification Model Comparison", ln=True, align="C")

    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        pdf_safe(
            "This report presents a structured comparison of multiple machine learning models for "
            "phishing email detection. The evaluation is conducted under controlled conditions "
            "using a consistent dataset and standardized preprocessing pipeline."
        ),
    )

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Models Evaluated", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        pdf_safe(
            "- Naive Bayes, used as a probabilistic baseline model.\n"
            "- Linear Support Vector Machine (SVM), suitable for high dimensional text features.\n"
            "- Hybrid MLP-SVM, a two-stage model combining neural feature learning with SVM classification."
        ),
    )

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Evaluation Metrics", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        pdf_safe(
            "Performance is evaluated using Accuracy, Precision, Recall, and F1 Score. "
            "These metrics reflect overall correctness, phishing detection capability, "
            "and classification balance."
        ),
    )

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary Metrics", ln=True)

    pdf.set_font("Helvetica", "", 10)
    headers = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"]
    widths = [55, 30, 30, 30, 20]

    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()

    for _, r in df_metrics.iterrows():
        pdf.cell(widths[0], 7, r["Model"], border=1)
        pdf.cell(widths[1], 7, f'{r["Accuracy (%)"]:.2f}', border=1, align="C")
        pdf.cell(widths[2], 7, f'{r["Precision (%)"]:.2f}', border=1, align="C")
        pdf.cell(widths[3], 7, f'{r["Recall (%)"]:.2f}', border=1, align="C")
        pdf.cell(widths[4], 7, f'{r["F1 (%)"]:.2f}', border=1, align="C")
        pdf.ln()

    # CLUSTERED METRICS CHART
    if HAS_MPL:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Overall Performance Comparison", ln=True)

        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(
            0, 6,
            pdf_safe(
                "The clustered bar chart compares Accuracy, Precision, Recall, and F1 Score "
                "across all evaluated models, providing a consolidated view of performance."
            ),
        )

        img_path = plot_metrics_clustered_png(df_metrics)
        tmp_imgs.append(img_path)
        pdf.ln(2)
        pdf.image(img_path, w=170)

    # CONFUSION MATRIX PAGES
    for model in MODEL_ORDER:
        cm = metrics_dict[model]["cm"]
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, f"Confusion Matrix Analysis: {model}", ln=True)

        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(
            0, 6,
            pdf_safe(
                "The confusion matrix illustrates how emails are classified as safe or phishing. "
                "Each value represents the number of correct or incorrect predictions."
            ),
        )

        if HAS_MPL:
            img_path = plot_cm_png(cm, f"Confusion Matrix: {model}")
            tmp_imgs.append(img_path)
            pdf.ln(2)
            pdf.image(img_path, w=135)

        pdf.ln(3)
        pdf.set_font("Helvetica", "", 11)

        if model == "Naive Bayes":
            pdf.multi_cell(
                0, 5,
                pdf_safe(
                    "The Naive Bayes confusion matrix shows that 2,178 legitimate emails were correctly "
                    "classified as safe, while 1,377 phishing emails were correctly detected and blocked. "
                    "However, 89 phishing emails were misclassified as safe, representing a security risk "
                    "because these attacks could bypass the system and reach users. In addition, "
                    "86 legitimate emails were incorrectly flagged as phishing, which may affect usability "
                    "but does not compromise security. The main concern lies in the false negatives, as "
                    "missed phishing emails pose direct security threats."
                ),
            )

        elif model == "SVM":
            pdf.multi_cell(
                0, 5,
                pdf_safe(
                    "The SVM confusion matrix indicates improved performance, with 2,191 legitimate emails "
                    "correctly classified as safe and 1,428 phishing emails successfully detected and blocked. "
                    "Importantly, only 38 phishing emails were misclassified as safe, reducing the risk of "
                    "phishing attacks bypassing the system. Meanwhile, 73 legitimate emails were incorrectly "
                    "flagged as phishing, which may cause minor inconvenience but does not affect security. "
                    "Overall, the reduced number of false negatives shows stronger protection against phishing "
                    "compared to Naive Bayes."
                ),
            )

        else:
            pdf.multi_cell(
                0, 5,
                pdf_safe(
                    "The hybrid model’s confusion matrix demonstrates the strongest security performance, "
                    "with 2,177 legitimate emails correctly classified as safe and 1,436 phishing emails "
                    "correctly detected and blocked. Only 30 phishing emails were misclassified as safe, "
                    "indicating the lowest risk of phishing bypass among the evaluated models. Although "
                    "87 legitimate emails were incorrectly flagged as phishing, this primarily impacts "
                    "usability rather than security. Overall, the hybrid approach minimizes the most critical "
                    "risk, missed phishing emails, providing the highest level of protection."
                ),
            )

            pdf.ln(12)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 6, "Summary", ln=True)

            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(
                0, 5,
                pdf_safe(
                    "The confusion matrix results show that all models are capable of detecting phishing emails, "
                    "but the Support Vector Machine (SVM) model provides the most effective balance between "
                    "security and usability. Compared to Naive Bayes, SVM significantly reduces the number of "
                    "phishing emails misclassified as safe, lowering the risk of attack bypass. While the hybrid "
                    "model also performs well, SVM achieves strong phishing detection with fewer usability "
                    "tradeoffs, making it the most suitable model for practical deployment in this study."
                ),
            )

    for p in tmp_imgs:
        try:
            os.remove(p)
        except Exception:
            pass

    output = pdf.output(dest="S")
    return bytes(output) if isinstance(output, (bytes, bytearray)) else output.encode("latin1")


# =========================
# STREAMLIT UI (OPTION 2)
# =========================
st.title("Results and Charts")
st.caption("The charts display results from pre evaluated models. No training is executed on this page")

dfm = metrics_table(metrics_all)
chart_df = dfm.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"]]

# Decision first
best_model = dfm.sort_values("F1 (%)", ascending=False).iloc[0]

st.success(
    f"Recommended Model Based on Evaluation Results: **{best_model['Model']}**\n\n"
    "Based on the comparative evaluation conducted in this study, this model "
    "demonstrates the most balanced performance in terms of phishing detection "
    "and false alert control, as reflected by F1 score and recall."
)

# Evidence
st.subheader("Evaluation Metrics Summary")
st.dataframe(dfm, use_container_width=True)

st.subheader("Model Performance Comparison")
st.bar_chart(chart_df)

# =========================
# PDF DOWNLOAD
# =========================
pdf_bytes = build_pdf_report(dfm, metrics_all)

st.download_button(
    "Download PDF report",
    data=pdf_bytes,
    file_name="phishing_detection_report.pdf",
    mime="application/pdf",
    use_container_width=True,
)
