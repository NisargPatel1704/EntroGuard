"""
EntroGuard SOC Dashboard v2
============================
Run on Raspberry Pi:
    streamlit run streamlit_dashboard_v2.py

Then open on any device on the same hotspot:
    http://<PI_IP>:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="EntroGuard SOC",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #000000;
    color: #ffffff;
}
h1, h2, h3 { color: #00ffcc; }
.stTabs [data-baseweb="tab-list"] { gap: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡 EntroGuard SOC Platform v2")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.markdown("### ⚙ Controls")
hybrid_threshold = st.sidebar.slider("Hybrid Score Threshold", 0.1, 1.0, 0.40)
syn_threshold    = st.sidebar.slider("SYN Ratio Threshold",    0.1, 1.0, 0.70)
packet_threshold = st.sidebar.slider("Packet Rate Threshold",   10, 500, 100)
st.sidebar.markdown("---")
st.sidebar.markdown("📱 **Hotspot Mode** — 2.4GHz")
st.sidebar.markdown("All devices on same mobile hotspot")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_data():
    if not os.path.exists("metrics_log.csv"):
        return pd.DataFrame()
    df = pd.read_csv("metrics_log.csv", header=None)
    if df.shape[1] == 17:
        df.columns = [
            "timestamp", "packet_rate", "syn_ratio", "ack_ratio", "rst_ratio",
            "src_entropy", "port_entropy", "unique_src", "unique_ports",
            "avg_size", "iat_variance", "entropy_score", "ml_prob",
            "hybrid_score", "cpu", "ram", "anomaly"
        ]
    elif df.shape[1] == 18:
        df.columns = [
            "timestamp", "packet_rate", "syn_ratio", "ack_ratio", "rst_ratio",
            "src_entropy", "port_entropy", "unique_src", "unique_ports",
            "avg_size", "iat_variance", "entropy_score", "ml_prob",
            "hybrid_score", "cpu", "ram", "anomaly", "device_ip"
        ]
    else:
        return pd.DataFrame()
    return df.tail(100)

df = load_data()

if df.empty:
    st.warning("⏳ No data yet. Make sure pi_receiver.py is running and ESP32 is connected to hotspot.")
    st.stop()

latest          = df.iloc[-1]
total_anomalies = int(df["anomaly"].sum())
total_windows   = len(df)

# ─────────────────────────────────────────────
# STATUS BANNER
# ─────────────────────────────────────────────

if latest["anomaly"] == 1:
    st.error("🔴 ALERT — ANOMALOUS TRAFFIC DETECTED")
else:
    st.success("🟢 SYSTEM NORMAL")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "📉 Attack Analytics",
    "🖥 System Health", "📑 Evaluation"
])

# ═════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Packet Rate",    round(latest.packet_rate,  2))
    c2.metric("SYN Ratio",      round(latest.syn_ratio,    4))
    c3.metric("ML Probability", round(latest.ml_prob,      3))
    c4.metric("Hybrid Score",   round(latest.hybrid_score, 3))
    c5.metric("Anomalies",      total_anomalies)

    st.markdown("---")

    intensity, color = "LOW", "green"
    if latest.hybrid_score >= hybrid_threshold:
        intensity, color = "HIGH", "red"
    elif latest.hybrid_score >= 0.25:
        intensity, color = "MEDIUM", "orange"

    st.markdown(f"### 🔥 Threat Level: :{color}[{intensity}]")
    st.progress(float(min(latest.hybrid_score, 1.0)),
                text=f"Hybrid Score: {latest.hybrid_score:.3f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(df, y=["packet_rate","syn_ratio"],
                      title="Network Traffic", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.line(df, y=["ml_prob","hybrid_score","entropy_score"],
                       title="Detection Scores", template="plotly_dark")
        fig2.add_hline(y=hybrid_threshold, line_dash="dash",
                       line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recent Windows")
    st.dataframe(df.tail(10), use_container_width=True)

# ═════════════════════════════════════════════
# TAB 2 — ATTACK ANALYTICS
# ═════════════════════════════════════════════
with tab2:
    ac = df["anomaly"].value_counts()
    fig_bar = px.bar(
        x=["Normal","Attack"],
        y=[ac.get(0,0), ac.get(1,0)],
        color=["Normal","Attack"],
        color_discrete_map={"Normal":"#00ffcc","Attack":"#ff4444"},
        template="plotly_dark", title="Normal vs Attack Windows"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.line(df, y=["src_entropy","port_entropy"],
                    title="Entropy Trends", template="plotly_dark"),
            use_container_width=True)
    with c2:
        st.plotly_chart(
            px.line(df, y=["syn_ratio","ack_ratio","rst_ratio"],
                    title="TCP Flag Ratios", template="plotly_dark"),
            use_container_width=True)

    st.plotly_chart(
        px.area(df, y="packet_rate",
                title="Packet Rate", template="plotly_dark"),
        use_container_width=True)

# ═════════════════════════════════════════════
# TAB 3 — SYSTEM HEALTH
# ═════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.line(df, y="cpu", title="Pi CPU Usage (%)", template="plotly_dark"),
            use_container_width=True)
    with c2:
        st.plotly_chart(
            px.line(df, y="ram", title="Pi RAM Usage (%)", template="plotly_dark"),
            use_container_width=True)

    ca, cb, cc = st.columns(3)
    ca.metric("Avg CPU",          f"{df['cpu'].mean():.1f}%")
    cb.metric("Avg RAM",          f"{df['ram'].mean():.1f}%")
    cc.metric("Windows Analyzed", total_windows)

# ═════════════════════════════════════════════
# TAB 4 — EVALUATION (FIXED)
# ═════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Detection Evaluation")

    y_true = df["anomaly"].astype(int)
    scores = df["hybrid_score"]
    y_pred = (scores >= hybrid_threshold).astype(int)  # ← fixed bug

    if y_true.nunique() < 2:
        st.warning("Need both normal and attack windows for evaluation metrics.")
    else:
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc     = auc(fpr, tpr)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy",  round(acc,  3))
        c2.metric("Precision", round(prec, 3))
        c3.metric("Recall",    round(rec,  3))
        c4.metric("F1 Score",  round(f1,   3))
        c5.metric("AUC",       round(roc_auc, 3))

        st.markdown("---")
        rc1, rc2 = st.columns(2)

        with rc1:
            fig_roc, ax = plt.subplots(facecolor="#111111")
            ax.set_facecolor("#111111")
            ax.plot(fpr, tpr, color="#00ffcc", lw=2, label=f"AUC={roc_auc:.3f}")
            ax.plot([0,1],[0,1],"--",color="#555555")
            ax.set_xlabel("FPR",color="white"); ax.set_ylabel("TPR",color="white")
            ax.set_title("ROC Curve",color="#00ffcc")
            ax.legend(facecolor="#222222",labelcolor="white")
            ax.tick_params(colors="white")
            st.pyplot(fig_roc)

        with rc2:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax = plt.subplots(facecolor="#111111")
            ax.set_facecolor("#111111")
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=["Normal","Attack"],
                        yticklabels=["Normal","Attack"], ax=ax)
            ax.set_title("Confusion Matrix",color="#00ffcc")
            ax.tick_params(colors="white")
            st.pyplot(fig_cm)

        st.markdown("---")
        if st.button("📄 Generate Security Report"):
            styles  = getSampleStyleSheet()
            report  = SimpleDocTemplate("entroguard_report.pdf")
            content = []
            content.append(Paragraph("EntroGuard Security Detection Report", styles["Title"]))
            content.append(Spacer(1, 20))
            content.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
            content.append(Spacer(1, 12))
            content.append(Paragraph("Traffic Summary", styles["Heading2"]))
            content.append(Paragraph(f"Total Windows : {total_windows}", styles["Normal"]))
            content.append(Paragraph(f"Anomalies     : {total_anomalies}", styles["Normal"]))
            content.append(Spacer(1, 12))
            content.append(Paragraph("Performance Metrics", styles["Heading2"]))
            data = [["Metric","Value"],
                    ["Accuracy",  f"{acc:.4f}"],
                    ["Precision", f"{prec:.4f}"],
                    ["Recall",    f"{rec:.4f}"],
                    ["F1 Score",  f"{f1:.4f}"],
                    ["AUC",       f"{roc_auc:.4f}"]]
            t = Table(data, colWidths=[200,100])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
                ("TEXTCOLOR", (0,0),(-1,0),colors.white),
                ("GRID",      (0,0),(-1,-1),0.5,colors.grey),
                ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
            ]))
            content.append(t)
            report.build(content)
            st.success("✅ Report saved: entroguard_report.pdf")
