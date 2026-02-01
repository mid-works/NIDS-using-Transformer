# NIDS-using-Transformer 🛡️🔥

**Hybrid Network Intrusion Detection System** powered by real-time packet capture, flow analysis, and **Transformer-based deep learning** for fast, accurate threat detection.


[![Python](https://img.shields.io/badge/python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/mid-works/NIDS-using-Transformer?style=for-the-badge&color=yellow)](https://github.com/mid-works/NIDS-using-Transformer/stargazers)
[![Forks](https://img.shields.io/github/forks/mid-works/NIDS-using-Transformer?style=for-the-badge&color=green)](https://github.com/mid-works/NIDS-using-Transformer/network/members)
[![Issues](https://img.shields.io/github/issues/mid-works/NIDS-using-Transformer?style=for-the-badge&color=red)](https://github.com/mid-works/NIDS-using-Transformer/issues)
[![Last Commit](https://img.shields.io/github/last-commit/mid-works/NIDS-using-Transformer?style=for-the-badge&color=purple)](https://github.com/mid-works/NIDS-using-Transformer/commits)

> [!IMPORTANT]  
> Requires **admin privileges** for live packet capture (Scapy needs it).  
> GPU strongly recommended for training.

## ✨ Features

- 🚀 **Real-time** packet capture & flow feature extraction (Scapy)
- 📊 Protocol dissection (TCP/UDP/ICMP) + flow timeout logic
- 🧠 **Transformer** deep learning model for anomaly & attack classification
- ⚡ Feature correlation + selection + scaling pipeline
- 📈 Beautiful **GUI dashboard** — live packet rate, alerts, history, severity
- 🔔 Probability-based multi-threshold alerting
- 🛠️ Fully modular — easy to swap models, datasets, thresholds

## 🚀 Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/mid-works/NIDS-using-Transformer.git
cd NIDS-using-Transformer

# 2. Virtual env (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Run (uses pre-trained model by default)
python main.py
```

Select network interface in GUI → Click Start Monitoring → Watch threats in real time!
[!TIP]
```bash
Want to retrain on your data?
python main.py --retrain
```

flowchart TD
```mermaid```
    A[Internet / Live Traffic] -->|Packets| B[Scapy Packet Capture]
    B --> C[Flow Aggregation & Feature Extraction]
    C --> D{Timeout?}
    D -->|Yes| E[Export Flow Features]
    E --> F[Preprocessing: Scaling + Encoding + Correlation Selection]
    F --> G[Transformer Model Inference]
    G --> H{Anomaly Probability > Threshold?}
    H -->|Yes| I[Generate Alert + Severity]
    I --> J[GUI Dashboard: Graphs, Alerts, History]
    J --> K[User: Real-time View + Logs]
    subgraph "Offline Training"
        L[UNSW-NB15 or Custom CSV] --> M[Train Transformer]
        M --> N[Save Best Model]
    end
    N --> G

   # ⚙️ Installation
Prerequisites

Python 3.12+
Admin / sudo rights (for live capture)
Optional: NVIDIA GPU + CUDA (for faster training)
```bash

git clone https://github.com/mid-works/NIDS-using-Transformer.git
cd NIDS-using-Transformer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
📊 Dataset
Uses UNSW-NB15 by default — modern, realistic network flows with 9 attack families.

  UNSW-NB15 Feature Importance Example
  
Example: Feature importance ranking from research on UNSW-NB15

Custom dataset?
Just place CSV in data/ with at least these columns:

```csv
dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sload,dload,label
```

🧠 Training
Bash# Retrain from scratch (takes time — use GPU!)
```
python main.py --retrain

# Or just use the saved model (fast)
python main.py

```

[!NOTE]
Tune hyperparameters in CONFIG dict: learning rate, epochs, flow timeout, alert thresholds, top-k features, etc.
```text
Example training log:
text[2025-02-01] Training Transformer...
Epoch 1/10 - loss: 0.685 → acc: 0.74
Epoch 2/10 - loss: 0.421 → acc: 0.89
```

📈 Real-Time Monitoring

Launch → GUI opens
Pick network interface (eth0, wlan0, etc.)
Hit Start
Watch:
Live packets/sec graph
Attack types pie/bar
Alert feed with severity (low/medium/high/critical)
History log


[!WARNING]
On Windows, Npcap must be installed for Scapy to capture packets.
🤝 Contributing
Love to have your help!

Fork & create branch (git checkout -b feature/new-attack-type)
Commit (git commit -m 'Add support for XYZ attack')
Push & open PR

Even docs, bug reports, or new dataset tests are super welcome!
📄 License
MIT © Midhun (mid-works) 2025–2026
Star ⭐ if this helps your security project!
