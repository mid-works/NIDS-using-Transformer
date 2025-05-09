# NIDS-using-Transformer


A hybrid network intrusion detection system combining packet capture, flow analysis, and Transformer-based deep learning for real-time threat detection.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training Models](#training-models)
- [License](#license)

## Features ✨

- **Real-Time Flow Monitoring**  
  - Packet capture and flow feature extraction
  - Protocol analysis (TCP/UDP/ICMP)
  - Flow timeout management
- **Advanced Detection Engine**  
  - Transformer-based neural network for anomaly detection
  - Feature correlation analysis
  - Probability-based alert thresholds
- **Comprehensive GUI**  
  - Real-time packet rate visualization
  - Attack classification dashboard
  - Alert history with severity levels
- **Modular Design**  
  - Preprocessing pipeline with feature scaling
  - Feature selection based on correlation
  - Model training and evaluation framework

## Quick Start

```bash
git clone https://github.com/yourusername/transformer-nids.git
cd transformer-nids
pip install -r requirements.txt
python main.py
```

## Architecture 🏗️

1. **Packet Capture Layer**: Scapy-based real-time packet sniffer
2. **Flow Analysis**: Feature extraction and flow timeout handling
3. **Preprocessing**: Feature scaling and categorical encoding
4. **Detection Engine**: Transformer model for classification
5. **Alert System**: Multi-threshold alert generation
6. **GUI**: Real-time monitoring dashboard

## Installation ⚙️

### Prerequisites
- Python 3.12+
- Administrative privileges for packet capture
- NVIDIA GPU (recommended for training)

```bash
# Clone repository
git clone https://github.com/mid-works/NIDS-using-Transformer
cd NIDS-using-Transformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage 🚀

### Data Preparation
Place your dataset in `data/` directory:
- `UNSW_NB15_training-set.csv` - Network flow data with labels

### Training the Model
```bash
# Train with default settings
python main.py --retrain

# Use pre-trained model (default)
python main.py
```

### Real-Time Monitoring
1. Select network interface from GUI dropdown
2. Click "Start Monitoring"
3. View real-time statistics and alerts

## Dataset 📊

The system uses the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) by default. To use custom data:

Required columns in CSV:
```csv
dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sload,dload,label
```

## Training Models 🧠

Configuration options in `CONFIG` dictionary:
- Model hyperparameters
- Flow timeout settings
- Alert thresholds
- Feature selection count

Example training output:
```
[2023-11-15 10:00:00] Training Transformer...
Epoch 1/5 - loss: 0.6921 - accuracy: 0.7124
Epoch 2/5 - loss: 0.5123 - accuracy: 0.8231
```

## Contributing 🤝

1. Fork the project
2. Create your feature branch 
3. Commit changes 
4. Push to branch 
5. Open a Pull Request

## License 📄

Distributed under the MIT License. See `LICENSE` for details.
