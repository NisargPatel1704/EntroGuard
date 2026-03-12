"""
EntroGuard — Raspberry Pi UDP Receiver
========================================
Listens for feature vectors from ESP32 over UDP,
runs hybrid XGBoost + entropy detection,
logs results to CSV for the Streamlit dashboard.

Run on Pi AFTER deploying with deploy.sh:
    python3 pi_receiver.py

NOTE: Pi and ESP32 must be on the same mobile hotspot!
"""

import socket
import json
import joblib
import pandas as pd
import numpy as np
import csv
import psutil
import signal
import sys
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

UDP_IP             = "0.0.0.0"        # listen on all interfaces
UDP_PORT           = 5005             # must match ESP32 PI_PORT
MODEL_PATH         = "xgb_model.pkl"
SCALER_PATH        = "scaler.pkl"

ENTROPY_WEIGHT     = 0.35
ML_WEIGHT          = 0.65
DECISION_THRESHOLD = 0.40             # lower = more sensitive

SRC_ENTROPY_MIN    = 1.0
PORT_ENTROPY_MIN   = 0.8

FEATURE_COLS = [
    "packet_rate", "syn_ratio", "ack_ratio", "rst_ratio",
    "src_ip_entropy", "dst_port_entropy", "unique_src_count",
    "unique_port_count", "avg_packet_size", "iat_variance"
]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  EntroGuard Pi Receiver — XGBoost Hybrid IDS")
print("  Mobile Hotspot Edition")
print("=" * 55)

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"\n✅ Model loaded: {MODEL_PATH}")
    print(f"✅ Scaler loaded: {SCALER_PATH}")
except FileNotFoundError as e:
    print(f"\n❌ {e}")
    print("\n  Run on your laptop first:")
    print("    cd model/ && python train_model.py")
    print("    bash deploy.sh")
    sys.exit(1)

# ─────────────────────────────────────────────
# UDP SOCKET
# ─────────────────────────────────────────────

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(30)

print(f"\n📡 Listening on UDP port {UDP_PORT}...")
print(f"   Threshold  : {DECISION_THRESHOLD}")
print(f"   ML weight  : {ML_WEIGHT}  |  Entropy weight: {ENTROPY_WEIGHT}")
print("\n  Waiting for ESP32 data (Ctrl+C to stop)...\n")

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

running         = True
total_windows   = 0
total_anomalies = 0

def shutdown(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, shutdown)

# ─────────────────────────────────────────────
# ENTROPY ANOMALY SCORE
# ─────────────────────────────────────────────

def entropy_anomaly_score(src_ent, port_ent, syn_ratio, pkt_rate):
    score = 0.0
    if src_ent  < SRC_ENTROPY_MIN:   score += 0.4   # few source IPs
    if port_ent < PORT_ENTROPY_MIN:  score += 0.3   # targeting single port
    if syn_ratio > 0.7:              score += 0.2   # flood of SYN packets
    if pkt_rate  > 200:              score += 0.1   # very high packet rate
    return min(score, 1.0)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def log_result(features, anomaly, hybrid_score,
               ml_prob, ent_score, cpu, ram, device_ip):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp] + features + [
        round(ent_score,    4),
        round(ml_prob,      4),
        round(hybrid_score, 4),
        round(cpu, 2),
        round(ram, 2),
        anomaly,
        device_ip
    ]
    with open("metrics_log.csv", "a", newline="") as f:
        csv.writer(f).writerow(row)
    if anomaly == 1:
        with open("anomaly_log.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)

# ─────────────────────────────────────────────
# PROCESS INCOMING UDP PACKET FROM ESP32
# ─────────────────────────────────────────────

def process_esp32_data(raw, sender_ip):
    global total_windows, total_anomalies

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"⚠ Bad JSON from {sender_ip}")
        return

    # Extract features
    try:
        fv = [
            float(data.get("packet_rate",       0)),
            float(data.get("syn_ratio",          0)),
            float(data.get("ack_ratio",          0)),
            float(data.get("rst_ratio",          0)),
            float(data.get("src_ip_entropy",     0)),
            float(data.get("dst_port_entropy",   0)),
            float(data.get("unique_src_count",   1)),
            float(data.get("unique_port_count",  0)),
            float(data.get("avg_packet_size",    0)),
            float(data.get("iat_variance",       0)),
        ]
    except (ValueError, TypeError) as e:
        print(f"⚠ Feature error: {e}")
        return

    device_ip  = data.get("device_ip", sender_ip)
    window_num = data.get("window",    0)
    rssi       = data.get("rssi",      0)

    # Scale + predict
    feat_df     = pd.DataFrame([fv], columns=FEATURE_COLS)
    feat_scaled = scaler.transform(feat_df)
    ml_prob     = float(model.predict_proba(feat_scaled)[0][1])

    # Entropy score + hybrid
    ent_score    = entropy_anomaly_score(fv[4], fv[5], fv[1], fv[0])
    hybrid_score = (ML_WEIGHT * ml_prob) + (ENTROPY_WEIGHT * ent_score)
    anomaly      = 1 if hybrid_score >= DECISION_THRESHOLD else 0

    # Pi system stats
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    total_windows += 1
    if anomaly == 1:
        total_anomalies += 1

    # Console output
    print(f"\n{'='*55}")
    print(f"  📡 ESP32 [{device_ip}]  Window #{window_num}  RSSI:{rssi}dBm")
    print(f"  Pi → CPU:{cpu}%  RAM:{ram}%")
    print(f"{'─'*55}")
    print(f"  Packet Rate     : {fv[0]:.2f} pkt/s")
    print(f"  SYN/ACK/RST     : {fv[1]:.3f} / {fv[2]:.3f} / {fv[3]:.3f}")
    print(f"  Src Entropy     : {fv[4]:.4f}   Port Entropy: {fv[5]:.4f}")
    print(f"  Unique Ports    : {int(fv[7])}   Avg Size: {fv[8]:.0f}B")
    print(f"  IAT Variance    : {fv[9]:.6f}")
    print(f"{'─'*55}")
    print(f"  ML Prob         : {ml_prob:.4f}")
    print(f"  Entropy Score   : {ent_score:.4f}")
    print(f"  Hybrid Score    : {hybrid_score:.4f}  (thresh:{DECISION_THRESHOLD})")
    print(f"{'─'*55}")

    if anomaly == 1:
        sev = "CRITICAL" if hybrid_score > 0.75 else "WARNING"
        print(f"  🚨 [{sev}] ANOMALY — {device_ip}")
    else:
        print(f"  ✅ NORMAL")

    print(f"  Session: {total_anomalies}/{total_windows} flagged")

    log_result(fv, anomaly, hybrid_score, ml_prob,
               ent_score, cpu, ram, device_ip)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

while running:
    try:
        data, addr = sock.recvfrom(1024)
        process_esp32_data(data.decode("utf-8"), addr[0])
    except socket.timeout:
        print("⏳ No data in 30s — is ESP32 connected to hotspot?")
    except Exception as e:
        if running:
            print(f"⚠ Error: {e}")

sock.close()
print(f"\n{'='*55}")
print(f"  Session: {total_windows} windows | {total_anomalies} anomalies")
print(f"{'='*55}")
print("Shutdown complete.")
