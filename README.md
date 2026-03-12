# 🛡 EntroGuard — Entropy-Based Hybrid Intrusion Detection System

> A real-time network anomaly detection system built on **ESP32 + Raspberry Pi 4**,
> using a hybrid of **Shannon Entropy Analysis** and **XGBoost Machine Learning**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Arduino](https://img.shields.io/badge/Arduino-ESP32-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Project Overview

EntroGuard is a lightweight Edge AI Intrusion Detection System (IDS) designed for
resource-constrained environments. It detects network anomalies such as:

- 🔴 DoS / DDoS Attacks
- 🔴 Port Scanning
- 🔴 Brute Force Attacks
- 🔴 Abnormal Traffic Patterns

---

## 🏗 System Architecture

```
Mobile Hotspot (2.4GHz)
        │
        ├─── ESP32 (Arduino C++)
        │       • Captures own network traffic
        │       • Extracts packet features
        │       • Sends via UDP every 5 seconds
        │
        └─── Raspberry Pi 4
                • Receives UDP from ESP32
                • Runs Hybrid Detection Engine
                  (XGBoost + Entropy Analysis)
                • Logs results to CSV
                • Serves Streamlit Dashboard

Laptop
  • Trains XGBoost model
  • Deploys model to Pi via SCP
```

---

## 📁 Project Structure

```
EntroGuard/
├── esp32/
│   └── esp32_forwarder.ino       # ESP32 Arduino code
│
├── pi/
│   └── pi_receiver.py            # Pi UDP receiver + detection engine
│
├── dashboard/
│   └── streamlit_dashboard_v2.py # Streamlit SOC dashboard
│
├── model/
│   └── train_model.py            # XGBoost training script (run on laptop)
│
├── deploy.sh                     # One-click deploy to Pi
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Hardware Requirements

| Component | Details |
|---|---|
| ESP32 | Any ESP32 Dev Board (supports 2.4GHz WiFi only) |
| Raspberry Pi 4 | 2GB RAM or higher recommended |
| Mobile Hotspot | **Must be set to 2.4GHz** (see setup below) |
| Laptop | For model training and deployment |

---

## 📱 Mobile Hotspot Setup (Important!)

ESP32 only supports **2.4GHz WiFi**. Configure your hotspot:

**iPhone:**
> Settings → Personal Hotspot → Enable **"Maximize Compatibility"**

**Android:**
> Settings → Hotspot → AP Band → Select **2.4GHz**

All three devices (ESP32, Pi, Laptop) must connect to the **same hotspot**.

---

## 🚀 Quick Start

### Step 1 — Configure ESP32
Open `esp32/esp32_forwarder.ino` and set:
```cpp
const char* WIFI_SSID     = "YOUR_HOTSPOT_NAME";
const char* WIFI_PASSWORD = "YOUR_HOTSPOT_PASSWORD";
const char* PI_IP         = "192.168.XXX.XXX"; // Pi's IP on hotspot
```
Install `ArduinoJson` via Arduino Library Manager, then flash to ESP32.

### Step 2 — Train Model (on Laptop)
```bash
pip install -r requirements.txt
cd model/
python train_model.py
```

### Step 3 — Deploy to Pi
```bash
# Edit deploy.sh and set PI_IP to your Pi's hotspot IP
nano deploy.sh

bash deploy.sh
```

### Step 4 — Run on Pi
```bash
# Terminal 1 — Detection Engine
python3 pi_receiver.py

# Terminal 2 — Dashboard
streamlit run streamlit_dashboard_v2.py
```

### Step 5 — Open Dashboard
```
http://<PI_IP>:8501
```

---

## 🔍 Finding Your Pi's IP

After connecting Pi to your mobile hotspot:
```bash
hostname -I
```
Or check your phone's hotspot connected devices list.

---

## 🧠 Detection Approach

### Hybrid Scoring Formula
```
Hybrid Score = (0.65 × ML Probability) + (0.35 × Entropy Score)

If Hybrid Score ≥ 0.40 → ANOMALY DETECTED
```

### Features Used
| Feature | Description |
|---|---|
| `packet_rate` | Packets per second |
| `syn_ratio` | Ratio of SYN packets |
| `ack_ratio` | Ratio of ACK packets |
| `rst_ratio` | Ratio of RST packets |
| `src_ip_entropy` | Shannon entropy of source IPs |
| `dst_port_entropy` | Shannon entropy of destination ports |
| `unique_src_count` | Number of unique source IPs |
| `unique_port_count` | Number of unique destination ports |
| `avg_packet_size` | Average packet size in bytes |
| `iat_variance` | Inter-arrival time variance |

### Attack Signatures Detected
| Attack | Key Indicators |
|---|---|
| DoS/DDoS | High packet rate + high SYN ratio + low src entropy |
| Port Scan | High port entropy + high RST ratio + low src entropy |
| Brute Force | Low port entropy + high SYN + moderate packet rate |

---

## 📊 Model Performance

| Metric | Isolation Forest (old) | XGBoost + SMOTE (new) |
|---|---|---|
| Accuracy | 0.930 | ~0.95+ |
| Precision | 0.800 | ~0.90+ |
| Recall | 0.400 ❌ | ~0.85+ ✅ |
| F1 Score | 0.533 | ~0.87+ |
| AUC | 0.707 | ~0.95+ |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Edge Device | ESP32 + Arduino IDE (C++) |
| Edge Computing | Raspberry Pi 4 + Python 3 |
| ML Model | XGBoost + SMOTE (imbalanced-learn) |
| Dashboard | Streamlit + Plotly |
| Communication | UDP over WiFi (2.4GHz mobile hotspot) |
| Dataset | Synthetic (CIC-IDS 2017 distribution) |

---

## 📦 Dependencies

```
xgboost
scikit-learn
imbalanced-learn
pandas
numpy
joblib
streamlit
plotly
psutil
seaborn
matplotlib
reportlab
```

---

## 👨‍💻 Author

**Nisarg Patel** - March 2026

---

## 📄 License

MIT License — free to use for academic and research purposes.
