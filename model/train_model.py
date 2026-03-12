"""
EntroGuard — Model Training Script
====================================
Run this on your LAPTOP before deploying to Pi.

Generates synthetic CIC-IDS style traffic data,
trains XGBoost with SMOTE, saves model + scaler.

Usage:
    pip install -r requirements.txt
    python train_model.py

Outputs:
    xgb_model.pkl   ← deploy this to Pi
    scaler.pkl      ← deploy this to Pi
    training_results.png
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
#    Mimics CIC-IDS 2017 feature distributions
# ─────────────────────────────────────────────

def generate_normal_traffic(n=2000):
    return pd.DataFrame({
        "packet_rate":       np.random.normal(30, 10, n).clip(1, 100),
        "syn_ratio":         np.random.beta(1.5, 8, n),
        "ack_ratio":         np.random.beta(6, 2, n),
        "rst_ratio":         np.random.beta(1, 15, n),
        "src_ip_entropy":    np.random.normal(2.8, 0.4, n).clip(0, 5),
        "dst_port_entropy":  np.random.normal(2.5, 0.5, n).clip(0, 5),
        "unique_src_count":  np.random.randint(2, 20, n).astype(float),
        "unique_port_count": np.random.randint(3, 25, n).astype(float),
        "avg_packet_size":   np.random.normal(500, 150, n).clip(40, 1500),
        "iat_variance":      np.random.exponential(0.002, n),
        "label": 0
    })

def generate_dos_traffic(n=300):
    """DoS/DDoS — high packet rate, high SYN, low entropy"""
    return pd.DataFrame({
        "packet_rate":       np.random.normal(350, 80, n).clip(100, 600),
        "syn_ratio":         np.random.beta(9, 1, n),
        "ack_ratio":         np.random.beta(1, 6, n),
        "rst_ratio":         np.random.beta(2, 5, n),
        "src_ip_entropy":    np.random.normal(0.8, 0.3, n).clip(0, 3),
        "dst_port_entropy":  np.random.normal(0.5, 0.2, n).clip(0, 2),
        "unique_src_count":  np.random.randint(1, 5, n).astype(float),
        "unique_port_count": np.random.randint(1, 4, n).astype(float),
        "avg_packet_size":   np.random.normal(60, 20, n).clip(40, 200),
        "iat_variance":      np.random.exponential(0.00005, n),
        "label": 1
    })

def generate_portscan_traffic(n=250):
    """Port scan — many unique ports, high RST, single source"""
    return pd.DataFrame({
        "packet_rate":       np.random.normal(15, 5, n).clip(1, 40),
        "syn_ratio":         np.random.beta(7, 2, n),
        "ack_ratio":         np.random.beta(1, 8, n),
        "rst_ratio":         np.random.beta(6, 2, n),
        "src_ip_entropy":    np.random.normal(0.3, 0.2, n).clip(0, 1),
        "dst_port_entropy":  np.random.normal(4.5, 0.3, n).clip(3, 5),
        "unique_src_count":  np.random.randint(1, 3, n).astype(float),
        "unique_port_count": np.random.randint(50, 200, n).astype(float),
        "avg_packet_size":   np.random.normal(60, 10, n).clip(40, 100),
        "iat_variance":      np.random.exponential(0.001, n),
        "label": 1
    })

def generate_bruteforce_traffic(n=200):
    """Brute force — single port, repeated SYN, moderate rate"""
    return pd.DataFrame({
        "packet_rate":       np.random.normal(80, 20, n).clip(30, 200),
        "syn_ratio":         np.random.beta(5, 3, n),
        "ack_ratio":         np.random.beta(4, 4, n),
        "rst_ratio":         np.random.beta(3, 4, n),
        "src_ip_entropy":    np.random.normal(0.5, 0.3, n).clip(0, 2),
        "dst_port_entropy":  np.random.normal(0.2, 0.1, n).clip(0, 0.5),
        "unique_src_count":  np.random.randint(1, 4, n).astype(float),
        "unique_port_count": np.random.randint(1, 3, n).astype(float),
        "avg_packet_size":   np.random.normal(200, 50, n).clip(40, 500),
        "iat_variance":      np.random.exponential(0.0005, n),
        "label": 1
    })

FEATURE_COLS = [
    "packet_rate", "syn_ratio", "ack_ratio", "rst_ratio",
    "src_ip_entropy", "dst_port_entropy", "unique_src_count",
    "unique_port_count", "avg_packet_size", "iat_variance"
]

print("=" * 55)
print("  EntroGuard — Model Training Pipeline")
print("=" * 55)

# ── Generate data ─────────────────────────────
print("\n[1/6] Generating synthetic traffic data...")
df = pd.concat([
    generate_normal_traffic(2000),
    generate_dos_traffic(300),
    generate_portscan_traffic(250),
    generate_bruteforce_traffic(200),
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"    Total    : {len(df)} samples")
print(f"    Normal   : {(df.label==0).sum()}")
print(f"    Attack   : {(df.label==1).sum()}")

# ── Split ─────────────────────────────────────
print("\n[2/6] Train/Test split (80/20 stratified)...")
X = df[FEATURE_COLS]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Scale ─────────────────────────────────────
print("\n[3/6] Scaling features...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── SMOTE ─────────────────────────────────────
print("\n[4/6] Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"    Before: Normal={( y_train==0).sum()} Attack={(y_train==1).sum()}")
print(f"    After : Normal={(y_resampled==0).sum()} Attack={(y_resampled==1).sum()}")

# ── Train XGBoost ─────────────────────────────
print("\n[5/6] Training XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_resampled, y_resampled,
          eval_set=[(X_test_scaled, y_test)], verbose=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring="f1")
print(f"    CV F1: {[round(s,3) for s in cv_f1]}  Mean={cv_f1.mean():.3f}")

# ── Evaluate ──────────────────────────────────
print("\n[6/6] Evaluating on test set...")
y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"  AUC       : {roc_auc_score(y_test, y_pred_proba):.4f}")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Normal","Attack"]))

# ── Save ──────────────────────────────────────
joblib.dump(model,  "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Saved: xgb_model.pkl")
print("✅ Saved: scaler.pkl")
print("\nNow run: bash deploy.sh\n")

# ── Plot ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0a0a0a")
for ax in axes:
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("#00ffcc")
    for s in ax.spines.values(): s.set_edgecolor("#333333")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc     = roc_auc_score(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#00ffcc", lw=2, label=f"AUC={roc_auc:.3f}")
axes[0].plot([0,1],[0,1],"--",color="#555555")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve")
axes[0].legend(facecolor="#222222", labelcolor="white")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=axes[1], cmap="YlOrRd",
            xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
importances.plot(kind="barh", ax=axes[2], color="#00ffcc")
axes[2].set_title("Feature Importance")
axes[2].set_xlabel("Score")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150,
            bbox_inches="tight", facecolor="#0a0a0a")
print("✅ Saved: training_results.png")
plt.show()
