import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset
from collections import deque, defaultdict
import threading
import time
import logging
from datetime import datetime
import joblib
import warnings
import psutil
from scapy.all import sniff, IP, TCP, UDP, conf
import socket
import seaborn as sns
import os
import ctypes
import statistics

# --- Configuration Constants ---
CONFIG = {
    "DATA_PATH": "data/UNSW_NB15_training-set.csv",  # Updated to new dataset path
    "SCALER_PATH": "models/scaler.joblib",
    "FEATURE_PATH": "models/selected_features.joblib",
    "MODEL_PATH": "models/transformer_nids_flow.pth",
    "PROTO_CATEGORIES": ['tcp', 'udp', 'icmp', 'other'],
    "FLOW_TIMEOUT_SECONDS": 60,
    "FEATURE_GENERATION_INTERVAL_SECONDS": 5,
    "TOP_N_FEATURES": 20,
    "ALERT_THRESHOLDS": {'LOW': 0.5, 'MEDIUM': 0.75, 'HIGH': 0.9},
    "GUI_UPDATE_INTERVAL_MS": 1000,
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(threadName)s - %(message)s')
warnings.filterwarnings('ignore')

# --- Transformer Model Definition ---
class TransformerNIDS(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerNIDS, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch in model forward pass. Expected {self.input_dim}, got {x.shape[-1]}")
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# --- NIDS Preprocessor Class ---
class NIDSPreprocessor:
    def __init__(self, scaler_path=CONFIG["SCALER_PATH"], proto_categories=CONFIG["PROTO_CATEGORIES"]):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.proto_categories = proto_categories
        self.scaler_path = scaler_path

    def load_data(self, filepath):
        """Loads the new dataset."""
        try:
            # Define columns based on the new dataset structure
            columns = [
                'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
                'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
                'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports',
                'attack_cat', 'label'
            ]
            df = pd.read_csv(filepath, names=columns, low_memory=False, skiprows=1)
            logging.info(f"Dataset loaded with shape: {df.shape}")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=['label'])
            return df
        except FileNotFoundError:
            logging.error(f"Dataset file not found: { filepath}")
            raise
        except Exception as e:
            logging.error(f"Error loading dataset from {filepath}: {e}")
            raise

    def _preprocess_features(self, df, is_training=True):
        """Preprocess features for the new dataset."""
        df = df.copy()

        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

        # Drop columns not used for modeling
        cols_to_drop = ['attack_cat']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

        # Protocol Encoding
        if 'proto' in df.columns:
            df['proto'] = df['proto'].str.lower().str.strip()
            current_categories = list(df['proto'].unique())
            all_categories = sorted(list(set(self.proto_categories + current_categories)))
            if 'other' not in all_categories:
                all_categories.append('other')
            proto_type = CategoricalDtype(categories=all_categories, ordered=False)
            try:
                df['proto'] = df['proto'].astype(proto_type).cat.codes
            except ValueError as e:
                logging.warning(f"Proto encoding issue: {e}. Mapping unknowns to 'other' code.")
                other_code = proto_type.categories.get_loc('other') if 'other' in proto_type.categories else -1
                df['proto'] = df['proto'].astype(proto_type).cat.codes.replace(-1, other_code)

        # Service and State Encoding
        for col in ['service', 'state']:
            if col in df.columns:
                df[col] = df[col].str.lower().str.strip()
                categories = list(df[col].unique())
                if '-' not in categories:
                    categories.append('-')
                cat_type = CategoricalDtype(categories=categories, ordered=False)
                try:
                    df[col] = df[col].astype(cat_type).cat.codes
                except ValueError as e:
                    logging.warning(f"{col} encoding issue: {e}. Mapping unknowns to -1.")
                    df[col] = df[col].astype(cat_type).cat.codes.replace(-1, -1)

        # Numeric Conversion
        numeric_cols_list = []
        for column in df.columns:
            if column not in ['label', 'proto', 'service', 'state']:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    original_non_numeric = df[column].apply(type).eq(str).sum()
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                        converted_nan = df[column].isna().sum()
                        if is_training and original_non_numeric > 0 and converted_nan > 0:
                            logging.debug(f"Column '{column}': Coerced {converted_nan}/{original_non_numeric} non-numeric entries to NaN.")
                        numeric_cols_list.append(column)
                    except Exception as e:
                        logging.warning(f"Could not convert column '{column}' to numeric: {e}. Setting to 0.")
                        df[column] = 0
                        if column not in numeric_cols_list:
                            numeric_cols_list.append(column)
                elif column not in numeric_cols_list:
                    numeric_cols_list.append(column)

        df = df.fillna(0)

        # Scaling
        numeric_cols_for_scaling = df.select_dtypes(include=np.number).columns
        numeric_cols_for_scaling = numeric_cols_for_scaling.difference(['label'])

        if len(numeric_cols_for_scaling) > 0:
            if is_training:
                logging.info(f"Fitting scaler on {len(numeric_cols_for_scaling)} columns.")
                self.scaler.fit(df[numeric_cols_for_scaling])
                os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
                joblib.dump(self.scaler, self.scaler_path)
                logging.info(f"Scaler fitted and saved to {self.scaler_path}")

            try:
                if not is_training and not hasattr(self.scaler, 'mean_'):
                    self.scaler = joblib.load(self.scaler_path)
                df[numeric_cols_for_scaling] = self.scaler.transform(df[numeric_cols_for_scaling])
            except FileNotFoundError:
                logging.error(f"Scaler file not found at {self.scaler_path}. Run training first.")
                raise
            except Exception as e:
                logging.error(f"Error applying scaler: {e}")
                raise
        else:
            logging.warning("No numeric columns found for scaling.")

        return df

    def fit_transform(self, df):
        """Fits the scaler and transforms the training data."""
        logging.info("Preprocessing and fitting scaler on training data...")
        processed_df = self._preprocess_features(df, is_training=True)
        self.feature_names = processed_df.drop('label', axis=1, errors='ignore').columns.tolist()
        logging.info(f"Preprocessor fitted. Feature names ({len(self.feature_names)}): {self.feature_names}")
        return processed_df

    def transform(self, flow_features_list):
        """Transforms a list of real-time flow feature dictionaries."""
        if not self.feature_names:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform on training data first.")

        if not flow_features_list:
            return pd.DataFrame(columns=self.feature_names + ['label'])

        df = pd.DataFrame(flow_features_list)

        # Add missing feature columns and fill with 0
        for feature in self.feature_names:
            if feature not in df.columns:
                logging.debug(f"Feature '{feature}' missing in real-time data, adding as 0.")
                df[feature] = 0

        final_cols_order = self.feature_names + ['label']
        final_cols_order = [col for col in final_cols_order if col in df.columns]
        df = df[final_cols_order]

        processed_df = self._preprocess_features(df, is_training=False)
        return processed_df

# --- Feature Engineer Class ---
class FeatureEngineer:
    def __init__(self, feature_path=CONFIG["FEATURE_PATH"], top_n=CONFIG["TOP_N_FEATURES"]):
        self.selected_features = None
        self.feature_path = feature_path
        self.top_n = top_n

    def select_features(self, X, y, cache=True):
        """Select important features based on correlation."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            try:
                y = pd.Series(y, index=X.index)
            except:
                raise ValueError("Input y must be a pandas Series or convertible to one")

        cache_loaded = False
        if cache:
            try:
                self.selected_features = joblib.load(self.feature_path)
                logging.info(f"Loaded {len(self.selected_features)} selected features from cache: {self.feature_path}")
                self.selected_features = [f for f in self.selected_features if f in X.columns]
                if not self.selected_features:
                    logging.warning("Cached features file loaded, but none match current data columns. Recalculating.")
                else:
                    cache_loaded = True
            except FileNotFoundError:
                logging.info(f"Selected features cache not found at {self.feature_path}. Calculating...")
            except Exception as e:
                logging.error(f"Error loading selected features cache: {e}. Calculating...")

        if not cache_loaded:
            numeric_X = X.select_dtypes(include=np.number)
            if numeric_X.empty:
                logging.warning("No numeric features found in X for correlation calculation. Using all columns.")
                self.selected_features = X.columns.tolist()
            else:
                try:
                    correlations = numeric_X.corrwith(y).abs().dropna().sort_values(ascending=False)
                    self.selected_features = correlations.index[:self.top_n].tolist()
                    if not self.selected_features:
                        logging.warning("Correlation calculation resulted in no features. Using all numeric features.")
                        self.selected_features = numeric_X.columns.tolist()
                    else:
                        logging.info(f"Selected top {len(self.selected_features)} features based on correlation.")
                    if cache:
                        try:
                            os.makedirs(os.path.dirname(self.feature_path), exist_ok=True)
                            joblib.dump(self.selected_features, self.feature_path)
                            logging.info(f"Selected features cached to {self.feature_path}")
                        except Exception as e:
                            logging.error(f"Error caching selected features: {e}")
                except Exception as e:
                    logging.exception(f"Error during correlation calculation: {e}. Using all numeric features.")
                    self.selected_features = numeric_X.columns.tolist()

        self.selected_features = [f for f in self.selected_features if f in X.columns]
        return X[self.selected_features]

    def transform(self, X):
        """Selects the pre-defined features from new data X."""
        if self.selected_features is None:
            try:
                self.selected_features = joblib.load(self.feature_path)
                logging.info(f"Loaded {len(self.selected_features)} selected features from cache for transform.")
                self.selected_features = [f for f in self.selected_features if f in X.columns]
            except FileNotFoundError:
                logging.error("Selected features cache not found. FeatureEngineer must be fitted or cache loaded before transform.")
                raise RuntimeError("FeatureEngineer transform called before features were selected/loaded.")
            except Exception as e:
                logging.error(f"Error loading selected features cache for transform: {e}")
                raise

        available_selected = [f for f in self.selected_features if f in X.columns]
        missing_in_X = [f for f in self.selected_features if f not in X.columns]
        if missing_in_X:
            logging.warning(f"Features expected by engineer but missing in input data: {missing_in_X}. These columns will be skipped.")

        if not available_selected:
            logging.error("No selected features found in the input DataFrame for transform.")
            return pd.DataFrame(columns=self.selected_features)

        return X[available_selected]

# --- Alert System Class ---
class AlertSystem:
    def __init__(self):
        self.alert_levels = CONFIG["ALERT_THRESHOLDS"]
        self.alert_history = deque(maxlen=1000)

    def generate_alert(self, prediction, attack_probability, flow_feature_dict):
        """Generate alert based on prediction and probability."""
        if prediction == 1:
            if attack_probability >= self.alert_levels['HIGH']:
                severity = 'HIGH'
            elif attack_probability >= self.alert_levels['MEDIUM']:
                severity = 'MEDIUM'
            elif attack_probability >= self.alert_levels['LOW']:
                severity = 'LOW'
            else:
                return None

            alert = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'severity': severity,
                'probability': f"{attack_probability:.4f}",
                'protocol': flow_feature_dict.get('proto', 'unknown'),
                'duration': flow_feature_dict.get('dur', 0.0)
            }

            logging.warning(f"ALERT Generated: {alert}")
            self.alert_history.append(alert)
            return alert
        return None

    def get_recent_alerts(self, count=10):
        """Returns the most recent alerts."""
        return list(self.alert_history)[-count:]

# --- Real-Time Flow Monitor Class ---
class RealTimeFlowMonitor:
    def __init__(self, flow_timeout=CONFIG["FLOW_TIMEOUT_SECONDS"],
                 generation_interval=CONFIG["FEATURE_GENERATION_INTERVAL_SECONDS"]):
        self.flow_timeout = flow_timeout
        self.generation_interval = generation_interval
        self.active_flows = {}  # Key: (proto_num, flow_id), Value: Flow info dict
        self.processed_flow_features = deque(maxlen=5000)
        self.interfaces = self._get_network_interfaces()
        self.current_interface = None
        self.is_monitoring = False
        self.start_time = None
        self._capture_thread = None
        self._feature_thread = None
        self.total_packets_captured = 0
        self.total_flows_processed = 0
        self.protocol_map = {1: 'icmp', 6: 'tcp', 17: 'udp'}
        self.lock = threading.Lock()
        self.flow_id_counter = 0  # Simple counter for flow identification

    def _get_network_interfaces(self):
        """Get list of available network interfaces with IPv4 addresses."""
        interfaces = []
        try:
            all_interfaces = psutil.net_if_addrs()
            for iface, addrs in all_interfaces.items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        interfaces.append(iface)
                        break
        except Exception as e:
            logging.error(f"Error getting network interfaces using psutil: {e}")
        if not interfaces:
            logging.warning("Could not detect any network interfaces with IPv4 addresses.")
        return interfaces

    def _get_flow_key(self, packet):
        """Generates a flow key based on protocol and a unique flow ID."""
        if IP not in packet:
            return None

        proto = packet[IP].proto
        with self.lock:
            self.flow_id_counter += 1
            flow_id = self.flow_id_counter
        return (proto, flow_id)

    def _packet_callback(self, packet):
        """Processes each captured packet to update flow statistics."""
        self.total_packets_captured += 1
        now = time.time()
        flow_key = self._get_flow_key(packet)

        if flow_key:
            proto = packet[IP].proto
            pkt_len = len(packet)

            with self.lock:
                if flow_key not in self.active_flows:
                    self.active_flows[flow_key] = {
                        'proto': proto,
                        'start_time': now,
                        'last_time': now,
                        'pkt_times': [now],
                        'src_pkts': 0,
                        'dst_pkts': 0,
                        'src_bytes': 0,
                        'dst_bytes': 0,
                        'pkts_data': []
                    }

                flow = self.active_flows[flow_key]
                flow['last_time'] = now
                flow['pkt_times'].append(now)

                # Simplified direction logic (alternate packets as src/dst)
                if len(flow['pkt_times']) % 2 == 1:
                    flow['src_pkts'] += 1
                    flow['src_bytes'] += pkt_len
                else:
                    flow['dst_pkts'] += 1
                    flow['dst_bytes'] += pkt_len

    def _calculate_features(self, flow_data):
        """Calculates features compatible with the new dataset."""
        features = {}
        start_time = flow_data['start_time']
        last_time = flow_data['last_time']

        features['proto'] = self.protocol_map.get(flow_data['proto'], 'other')
        features['dur'] = max(0, last_time - start_time)
        features['spkts'] = flow_data['src_pkts']
        features['dpkts'] = flow_data['dst_pkts']
        features['sbytes'] = flow_data['src_bytes']
        features['dbytes'] = flow_data['dst_bytes']
        duration = features['dur']
        features['rate'] = (features['spkts'] + features['dpkts']) / duration if duration > 1e-6 else 0
        features['sload'] = (features['sbytes'] * 8) / duration if duration > 1e-6 else 0
        features['dload'] = (features['dbytes'] * 8) / duration if duration > 1e-6 else 0
        features['smean'] = features['sbytes'] / features['spkts'] if features['spkts'] > 0 else 0
        features['dmean'] = features['dbytes'] / features['dpkts'] if features['dpkts'] > 0 else 0

        pkt_times = sorted(flow_data['pkt_times'])
        intpkts = [pkt_times[i] - pkt_times[i-1] for i in range(1, len(pkt_times))]
        features['sinpkt'] = statistics.mean(intpkts) if len(intpkts) > 0 else 0
        features['dinpkt'] = features['sinpkt']
        features['sjit'] = statistics.stdev(intpkts) if len(intpkts) > 1 else 0
        features['djit'] = features['sjit']

        # Placeholder features
        placeholder_features = [
            'service', 'state', 'sloss', 'dloss', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
            'trans_depth', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'is_ftp_login', 'ct_ftp_cmd',
            'ct_flw_http_mthd', 'is_sm_ips_ports'
        ]
        for feat in placeholder_features:
            features[feat] = 0

        features['label'] = 0
        return features

    def _generate_features_loop(self):
        """Periodically checks for timed-out flows and generates their features."""
        while self.is_monitoring:
            now = time.time()
            timed_out_keys = []

            with self.lock:
                for key, flow_data in self.active_flows.items():
                    if now - flow_data['last_time'] > self.flow_timeout:
                        timed_out_keys.append(key)

                for key in timed_out_keys:
                    try:
                        flow_data = self.active_flows.pop(key)
                        if flow_data['src_pkts'] > 0 or flow_data['dst_pkts'] > 0:
                            flow_features = self._calculate_features(flow_data)
                            self.processed_flow_features.append(flow_features)
                            self.total_flows_processed += 1
                    except KeyError:
                        logging.warning(f"Flow key {key} not found during timeout processing.")
                    except Exception as e:
                        logging.exception(f"Error calculating features for flow {key}: {e}")

            time.sleep(max(0.1, self.generation_interval))

    def start_capture(self, interface=None):
        """Starts packet capture and feature generation threads."""
        if self.is_monitoring:
            logging.warning("Flow monitoring is already running")
            return
        if not interface or interface not in self.interfaces:
            available = ", ".join(self.interfaces) if self.interfaces else "None found"
            raise ValueError(f"Invalid or unspecified interface '{interface}'. Available: {available}")

        self.current_interface = interface
        self.is_monitoring = True
        self.start_time = datetime.now()
        with self.lock:
            self.active_flows.clear()
            self.processed_flow_features.clear()
        self.total_packets_captured = 0
        self.total_flows_processed = 0
        self.flow_id_counter = 0
        logging.info(f"Starting flow monitoring on interface: {self.current_interface}")

        def capture_thread_target():
            try:
                logging.info(f"Scapy sniff starting on {self.current_interface}")
                conf.iface = self.current_interface
                sniff(
                    iface=self.current_interface,
                    prn=self._packet_callback,
                    store=0,
                    stop_filter=lambda p: not self.is_monitoring,
                )
                logging.info(f"Scapy sniff stopped gracefully on {self.current_interface}")
            except PermissionError:
                logging.error("Permission denied to sniff. Run the script as root/administrator.")
                self.is_monitoring = False
            except OSError as e:
                logging.error(f"OS error during sniff setup on {self.current_interface}: {e}")
                self.is_monitoring = False
            except Exception as e:
                logging.exception(f"Unexpected error in packet capture thread: {e}")
                self.is_monitoring = False

        def feature_thread_target():
            try:
                self._generate_features_loop()
                logging.info("Feature generation loop finished.")
            except Exception as e:
                logging.exception(f"Unexpected error in feature generation thread: {e}")
                self.is_monitoring = False

        self._capture_thread = threading.Thread(target=capture_thread_target, name="CaptureThread", daemon=True)
        self._feature_thread = threading.Thread(target=feature_thread_target, name="FeatureThread", daemon=True)
        self._capture_thread.start()
        self._feature_thread.start()

    def stop_capture(self):
        """Stops the monitoring threads and processes remaining flows."""
        if not self.is_monitoring and not self._capture_thread and not self._feature_thread:
            logging.info("Flow monitoring is already stopped.")
            return

        logging.info("Stopping flow monitoring...")
        self.is_monitoring = False

        if self._capture_thread and self._capture_thread.is_alive():
            logging.debug("Waiting for capture thread to stop...")
            self._capture_thread.join(timeout=2)
            if self._capture_thread.is_alive():
                logging.warning("Capture thread did not stop within timeout.")

        if self._feature_thread and self._feature_thread.is_alive():
            logging.debug("Waiting for feature thread to stop...")
            self._feature_thread.join(timeout=self.generation_interval + 1)
            if self._feature_thread.is_alive():
                logging.warning("Feature generation thread did not stop within timeout.")

        logging.info("Threads stopped.")
        self._capture_thread = None
        self._feature_thread = None
        self._process_remaining_flows()
        logging.info("Flow monitoring stopped completely.")

    def _process_remaining_flows(self):
        """Processes flows still in the active_flows dict."""
        with self.lock:
            remaining_keys = list(self.active_flows.keys())
            if remaining_keys:
                logging.info(f"Processing {len(remaining_keys)} remaining active flows...")
                for key in remaining_keys:
                    try:
                        flow_data = self.active_flows.pop(key)
                        if flow_data['src_pkts'] > 0 or flow_data['dst_pkts'] > 0:
                            flow_features = self._calculate_features(flow_data)
                            self.processed_flow_features.append(flow_features)
                            self.total_flows_processed += 1
                    except KeyError:
                        continue
                    except Exception as e:
                        logging.exception(f"Error calculating features for remaining flow {key}: {e}")
                logging.info("Finished processing remaining flows.")

    def get_recent_flow_features(self):
        """Retrieves and removes processed flow features from the queue."""
        features_list = []
        with self.lock:
            while True:
                try:
                    features_list.append(self.processed_flow_features.popleft())
                except IndexError:
                    break
        return features_list

    def get_statistics(self):
        """Gets current monitoring statistics."""
        with self.lock:
            active_flows_count = len(self.active_flows)
        elapsed_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        rate = self.total_packets_captured / elapsed_time if elapsed_time > 1e-6 else 0
        return {
            'packets_captured': self.total_packets_captured,
            'flows_processed': self.total_flows_processed,
            'active_flows': active_flows_count,
            'packet_rate': rate,
            'current_interface': self.current_interface,
            'monitoring_time': elapsed_time,
            'is_active': self.is_monitoring
        }

# --- Real-Time Detector Class ---
class RealTimeDetector:
    def __init__(self, preprocessor, feature_engineer, model_path=CONFIG["MODEL_PATH"]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.model_path = model_path
        self.model = None
        self.input_dim = None
        self.alert_system = AlertSystem()
        logging.info(f"RealTimeDetector using device: {self.device}")

    def train(self, X, y):
        """Trains the Transformer model."""
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("Training requires preprocessed X (DataFrame) and y (Series)")
        logging.info("Selecting features for training...")
        X_selected = self.feature_engineer.select_features(X, y)
        self.input_dim = X_selected.shape[1]
        if self.input_dim == 0:
            raise ValueError("No features selected for training.")
        logging.info(f"Starting training with input dimension: {self.input_dim}")
        logging.info(f"Features used for training ({len(self.feature_engineer.selected_features)}): {self.feature_engineer.selected_features}")
        self.model = TransformerNIDS(input_dim=self.input_dim, num_classes=2).to(self.device)
        X_tensor = torch.tensor(X_selected.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        batch_size = 128
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = 5
        logging.info(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            batches_processed = 0
            for i, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.unsqueeze(1).to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches_processed += 1
                if (i + 1) % 100 == 0:
                    logging.debug(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Batch Loss: {loss.item():.4f}")
            avg_loss = epoch_loss / batches_processed
            logging.info(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'selected_features': self.feature_engineer.selected_features
        }, self.model_path)
        logging.info(f"Transformer model trained and saved to {self.model_path}")

    def load_model(self):
        """Loads the Transformer model state and associated info."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logging.info(f"Loading model checkpoint from {self.model_path}")
            self.input_dim = checkpoint['input_dim']
            loaded_features = checkpoint.get('selected_features')
            self.model = TransformerNIDS(input_dim=self.input_dim, num_classes=2).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            if loaded_features:
                self.feature_engineer.selected_features = loaded_features
                logging.info(f"Model loaded. Input dim: {self.input_dim}. Expecting features: {self.feature_engineer.selected_features}")
            else:
                logging.error("Model loaded, but selected features list not found in checkpoint!")
                if self.feature_engineer.selected_features is None:
                    raise RuntimeError("Loaded model missing feature list, and FeatureEngineer is not fitted.")
            return True
        except FileNotFoundError:
            logging.info(f"Model file not found at {self.model_path}. Training is required.")
            return False
        except Exception as e:
            logging.exception(f"Error loading model from {self.model_path}: {e}")
            return False

    def process_flow_features(self, flow_feature_dict):
        """Processes a single dictionary of calculated flow features for detection."""
        if not self.model:
            logging.error("Model not loaded or trained. Cannot process features.")
            return {'is_attack': False, 'probability': 0, 'alert': None}
        if not self.preprocessor.feature_names:
            logging.error("Preprocessor not fitted. Cannot process features.")
            return {'is_attack': False, 'probability': 0, 'alert': None}
        if not self.feature_engineer.selected_features:
            logging.error("FeatureEngineer not fitted/loaded. Cannot process features.")
            return {'is_attack': False, 'probability': 0, 'alert': None}

        try:
            processed_df = self.preprocessor.transform([flow_feature_dict])
            if processed_df.empty:
                logging.warning("Preprocessing returned empty DataFrame for the flow.")
                return {'is_attack': False, 'probability': 0, 'alert': None}
            X_flow = processed_df.drop('label', axis=1, errors='ignore')
            X_flow_selected = self.feature_engineer.transform(X_flow)
            if X_flow_selected.shape[1] != self.input_dim:
                logging.warning(f"Feature mismatch after selection: Expected {self.input_dim} ({self.feature_engineer.selected_features}), got {X_flow_selected.shape[1]} ({X_flow_selected.columns.tolist()}). Attempting fix.")
                expected_cols = self.feature_engineer.selected_features
                current_cols = X_flow_selected.columns
                for col in expected_cols:
                    if col not in current_cols:
                        X_flow_selected[col] = 0
                        logging.debug(f"Added missing feature '{col}' as 0.")
                try:
                    X_flow_selected = X_flow_selected[expected_cols]
                except KeyError as e:
                    logging.error(f"Could not reconcile feature dimensions even after adding zeros. Missing key: {e}")
                    raise ValueError("Could not reconcile feature dimensions for model prediction.") from e
                if X_flow_selected.shape[1] != self.input_dim:
                    raise ValueError(f"Feature dimension mismatch persists after fix. Expected {self.input_dim}, got {X_flow_selected.shape[1]}.")
                logging.info("Feature mismatch resolved by adding zero columns and reordering.")
            X_tensor = torch.tensor(X_flow_selected.values, dtype=torch.float32).to(self.device)
            X_tensor = X_tensor.unsqueeze(1)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                attack_probability = probabilities[0][1].item()
                prediction = torch.argmax(probabilities, dim=1).item()
            is_attack = prediction == 1
            alert = None
            if is_attack:
                alert = self.alert_system.generate_alert(prediction, attack_probability, flow_feature_dict)
            return {'is_attack': is_attack, 'probability': attack_probability, 'alert': alert}
        except Exception as e:
            logging.exception(f"Error processing flow features for detection: {e}")
            return {'is_attack': False, 'probability': 0, 'alert': None}

# --- NIDS Monitor GUI Class ---
class NIDSMonitorGUI:
    def __init__(self, detector, alert_system, flow_monitor):
        self.detector = detector
        self.alert_system = alert_system
        self.flow_monitor = flow_monitor
        self.total_flows_analyzed = 0
        self.attack_count = 0
        self.packet_rates = deque(maxlen=120)
        self.protocol_counts = defaultdict(int)
        self.is_monitoring = False
        self.gui_update_thread = None
        self.last_alert_check_time = time.time()
        self.root = tk.Tk()
        self.root.title("Real-Time NIDS - Flow Analysis Monitor")
        self.root.geometry("1200x850")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        style = ttk.Style()
        style.theme_use('clam')
        self.top_frame = ttk.Frame(self.root, padding="5")
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        self.stats_frame = ttk.LabelFrame(self.top_frame, text="Monitoring Statistics", padding="5")
        self.stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.status_frame = ttk.LabelFrame(self.top_frame, text="System Status", padding="5")
        self.status_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.graphs_frame = ttk.Frame(self.root, padding="5")
        self.graphs_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.alert_frame = ttk.LabelFrame(self.root, text="Recent Alerts", padding="5")
        self.alert_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.button_frame = ttk.Frame(self.root, padding="5")
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.flow_stats_label = ttk.Label(self.stats_frame, text="Flows: - | Packets: - | Rate: - pps", font=('Helvetica', 10))
        self.flow_stats_label.pack(side=tk.TOP, anchor='w', pady=2)
        self.active_flows_label = ttk.Label(self.stats_frame, text="Active Flows: - | Monitoring Time: 0.0s", font=('Helvetica', 10))
        self.active_flows_label.pack(side=tk.TOP, anchor='w', pady=2)
        self.protocol_stats_label = ttk.Label(self.stats_frame, text="Protocols: Initializing...", font=('Helvetica', 10))
        self.protocol_stats_label.pack(side=tk.TOP, anchor='w', pady=2)
        self.attack_info_label = ttk.Label(self.status_frame, text="Status: Idle", foreground="blue", font=('Helvetica', 14, 'bold'))
        self.attack_info_label.pack(pady=10, padx=10)
        plt.style.use('ggplot')
        self.fig1 = Figure(figsize=(7, 3.5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.graphs_frame)
        self.canvas1_widget = self.canvas1.get_tk_widget()
        self.canvas1_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.fig1.tight_layout()
        self.fig2 = Figure(figsize=(7, 3.5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graphs_frame)
        self.canvas2_widget = self.canvas2.get_tk_widget()
        self.canvas2_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.fig2.tight_layout()
        self._initialize_graphs()
        self.alert_text = tk.Text(self.alert_frame, height=6, width=100, state='disabled', wrap=tk.WORD, font=('Courier', 9))
        self.alert_scrollbar = ttk.Scrollbar(self.alert_frame, orient=tk.VERTICAL, command=self.alert_text.yview)
        self.alert_text.config(yscrollcommand=self.alert_scrollbar.set)
        self.alert_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alert_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.interface_label = ttk.Label(self.button_frame, text="Interface:")
        self.interface_label.pack(side=tk.LEFT, padx=(0, 5))
        self.interfaces = self.flow_monitor.interfaces
        self.interface_var = tk.StringVar()
        self.interface_combo = ttk.Combobox(self.button_frame, textvariable=self.interface_var, values=self.interfaces, state='readonly', width=15)
        if self.interfaces:
            self.interface_var.set(self.interfaces[0])
        self.interface_combo.pack(side=tk.LEFT, padx=5)
        self.start_button = ttk.Button(self.button_frame, text="Start Monitoring", command=self._start_monitoring, width=15)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(self.button_frame, text="Stop Monitoring", command=self._stop_monitoring, state='disabled', width=15)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def _initialize_graphs(self):
        """Sets up the initial appearance of the graphs."""
        self.ax1.set_title('Network Packet Rate')
        self.ax1.set_ylabel('Packets/second')
        self.ax1.set_xlabel('Time (Updates)')
        self.ax1.grid(True)
        self.rate_line, = self.ax1.plot([], [], 'b-', label='Packet Rate (pps)')
        self.ax1.legend(loc='upper left')
        self.ax2.set_title('Flow Classification')
        self.ax2.set_ylabel('Number of Flows')
        self.classification_bars = self.ax2.bar(['Normal', 'Attack'], [0, 0], color=['#64dd17', '#d50000'])
        self.ax2.grid(axis='y', linestyle='--')
        self.canvas1.draw()
        self.canvas2.draw()

    def _update_protocol_stats(self, flow_features):
        """Updates protocol counts based on flow data."""
        proto_name = flow_features.get('proto', 'other')
        self.protocol_counts[str(proto_name).lower()] += 1

    def _display_alert(self, alert_dict):
        """Adds a formatted alert string to the text box."""
        if alert_dict:
            alert_str = (f"[{alert_dict['timestamp']}] {alert_dict['severity']} "
                         f"(P={alert_dict['probability']}) - "
                         f"Protocol: {alert_dict['protocol']}, Duration: {alert_dict['duration']:.2f}s\n")
            self.alert_text.config(state='normal')
            self.alert_text.insert(tk.END, alert_str)
            self.alert_text.see(tk.END)
            self.alert_text.config(state='disabled')

    def _start_monitoring(self):
        """Handles the 'Start Monitoring' button click."""
        if self.is_monitoring:
            logging.warning("Monitoring is already active.")
            return
        selected_interface = self.interface_var.get()
        if not selected_interface:
            messagebox.showerror("Interface Error", "Please select a network interface.")
            return
        self.total_flows_analyzed = 0
        self.attack_count = 0
        self.packet_rates.clear()
        self.protocol_counts.clear()
        self.alert_text.config(state='normal')
        self.alert_text.delete(1.0, tk.END)
        self.alert_text.config(state='disabled')
        self._initialize_graphs()
        try:
            logging.info(f"Attempting to start monitoring on {selected_interface}")
            self.flow_monitor.start_capture(selected_interface)
            self.is_monitoring = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.interface_combo.config(state='disabled')
            self.attack_info_label.config(text="Status: Monitoring...", foreground="#0277bd")
            self.gui_update_thread = threading.Thread(target=self._gui_update_loop, name="GUIUpdateThread", daemon=True)
            self.gui_update_thread.start()
            logging.info(f"Monitoring started successfully on {selected_interface}")
        except (ValueError, PermissionError, Exception) as e:
            logging.exception(f"Failed to start monitoring on {selected_interface}")
            messagebox.showerror("Monitoring Start Error", f"Could not start monitoring:\n{str(e)}")
            self.is_monitoring = False
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.interface_combo.config(state='readonly')
            self.attack_info_label.config(text="Status: Error", foreground="#c62828")

    def _stop_monitoring(self):
        """Handles the 'Stop Monitoring' button click."""
        if not self.is_monitoring and not self.flow_monitor.is_monitoring:
            logging.info("Monitoring is already stopped.")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.interface_combo.config(state='readonly')
            return
        logging.info("Stop monitoring requested.")
        self.is_monitoring = False
        self.flow_monitor.stop_capture()
        if self.gui_update_thread and self.gui_update_thread.is_alive():
            logging.debug("Waiting for GUI update thread to finish...")
            self.gui_update_thread.join(timeout=max(1.5, CONFIG["GUI_UPDATE_INTERVAL_MS"] / 1000.0 * 2))
            if self.gui_update_thread.is_alive():
                logging.warning("GUI update thread did not finish gracefully.")
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.interface_combo.config(state='readonly')
        self.attack_info_label.config(text="Status: Stopped", foreground="#ff8f00")
        self._update_displays()
        logging.info("GUI monitoring stopped.")

    def _gui_update_loop(self):
        """The target function for the GUI update thread."""
        logging.info("GUI update loop thread started.")
        while self.is_monitoring:
            try:
                self.root.after(0, self._update_displays)
                time.sleep(CONFIG["GUI_UPDATE_INTERVAL_MS"] / 1000.0)
            except tk.TclError as e:
                logging.warning(f"GUI update loop interrupted (TclError): {e}. Assuming window closed.")
                self.is_monitoring = False
                break
            except Exception as e:
                logging.exception(f"Unexpected error in GUI update loop: {e}")
                time.sleep(1)
        logging.info("GUI update loop thread finished.")

    def _update_displays(self):
        """Updates all GUI labels and graphs."""
        if not self.root or not self.root.winfo_exists():
            logging.warning("Attempted to update displays, but root window doesn't exist.")
            self.is_monitoring = False
            return
        try:
            flow_features_list = self.flow_monitor.get_recent_flow_features()
            detected_attack_in_batch = False
            if flow_features_list:
                self.total_flows_analyzed += len(flow_features_list)
                for flow_features in flow_features_list:
                    self._update_protocol_stats(flow_features)
                    result = self.detector.process_flow_features(flow_features)
                    if result['is_attack']:
                        self.attack_count += 1
                        detected_attack_in_batch = True
                        if result['alert']:
                            self._display_alert(result['alert'])
            stats = self.flow_monitor.get_statistics()
            self.flow_stats_label.config(text=f"Flows: {stats['flows_processed']} | "
                                             f"Packets: {stats['packets_captured']} | "
                                             f"Rate: {stats['packet_rate']:.1f} pps")
            self.active_flows_label.config(text=f"Active Flows: {stats['active_flows']} | "
                                               f"Monitoring Time: {stats['monitoring_time']:.1f}s")
            proto_items = sorted(self.protocol_counts.items())
            proto_text = "Protocols: " + " | ".join([f"{p.upper()}: {c}" for p, c in proto_items])
            self.protocol_stats_label.config(text=proto_text)
            if self.flow_monitor.is_monitoring:
                if detected_attack_in_batch:
                    self.attack_info_label.config(text=f"Status: ATTACK DETECTED! ({self.attack_count} total)", foreground="#d50000")
                else:
                    self.attack_info_label.config(text=f"Status: Monitoring - OK ({self.attack_count} attacks)", foreground="#1b5e20")
            self._update_graphs(stats)
        except Exception as e:
            logging.exception(f"Error occurred during GUI display update: {e}")

    def _update_graphs(self, current_stats):
        """Updates the graphs with the latest data."""
        try:
            current_rate = current_stats.get('packet_rate', 0)
            self.packet_rates.append(current_rate)
            self.ax1.clear()
            rate_data = list(self.packet_rates)
            if rate_data:
                self.ax1.plot(range(len(rate_data)), rate_data, 'b-', label='Packet Rate (pps)')
            else:
                self.ax1.plot([], [], 'b-', label='Packet Rate (pps)')
            self.ax1.set_title('Network Packet Rate')
            self.ax1.set_ylabel('Packets/second')
            self.ax1.set_xlabel('Time (Updates)')
            self.ax1.grid(True)
            self.ax1.legend(loc='upper left')
            self.ax1.relim()
            self.ax1.autoscale_view(True, True, True)
            self.canvas1.draw_idle()
            self.ax2.clear()
            normal_flows = max(0, self.total_flows_analyzed - self.attack_count)
            attack_flows = self.attack_count
            labels = ['Normal', 'Attack']
            counts = [normal_flows, attack_flows]
            colors = ['#64dd17', '#d50000']
            self.ax2.bar(labels, counts, color=colors)
            self.ax2.set_title('Flow Classification')
            self.ax2.set_ylabel('Number of Flows Processed')
            self.ax2.grid(axis='y', linestyle='--')
            self.ax2.relim()
            self.ax2.autoscale_view(True, False, True)
            current_ylim = self.ax2.get_ylim()
            self.ax2.set_ylim(bottom=0, top=max(1, current_ylim[1] * 1.1))
            self.canvas2.draw_idle()
        except Exception as e:
            logging.exception(f"Error updating graphs: {e}")

    def _on_closing(self):
        """Handles the window close event."""
        if self.is_monitoring:
            if messagebox.askyesno("Quit Application?", "Monitoring is currently active. Stop monitoring and exit?"):
                logging.info("User initiated exit while monitoring.")
                self._stop_monitoring()
                self.root.destroy()
            else:
                logging.info("User cancelled exit.")
                return
        else:
            logging.info("Exiting application.")
            self.root.destroy()

    def run(self):
        """Starts the Tkinter main event loop."""
        logging.info("Starting NIDS Monitor GUI...")
        try:
            self.root.mainloop()
        except Exception as e:
            logging.exception(f"An error occurred in the GUI main loop: {e}")
        finally:
            logging.info("GUI main loop ended.")
            if self.flow_monitor.is_monitoring:
                logging.warning("GUI exited, but flow monitor might still be running. Attempting stop.")
                self._stop_monitoring()

# --- Main Execution Function ---
def main():
    """Initializes components and starts the NIDS application."""
    PERFORM_RETRAINING = True  # Force retraining for new dataset
    try:
        logging.info("Initializing NIDS components...")
        preprocessor = NIDSPreprocessor()
        feature_engineer = FeatureEngineer()
        alert_system = AlertSystem()
        detector = RealTimeDetector(preprocessor, feature_engineer)
        detector.alert_system = alert_system
        flow_monitor = RealTimeFlowMonitor()
        try:
            logging.info(f"Loading dataset for fitting preprocessor/engineer: {CONFIG['DATA_PATH']}")
            training_data = preprocessor.load_data(CONFIG['DATA_PATH'])
            logging.info("Fitting preprocessor...")
            processed_training_data = preprocessor.fit_transform(training_data)
            if 'label' not in processed_training_data.columns:
                raise ValueError("'label' column missing after preprocessing.")
            X_train = processed_training_data.drop('label', axis=1)
            y_train = processed_training_data['label']
            logging.info("Fitting feature engineer...")
            _ = feature_engineer.select_features(X_train, y_train)
            logging.info(f"Feature engineer fitted. Selected features ({len(feature_engineer.selected_features)}): {feature_engineer.selected_features}")
        except FileNotFoundError:
            logging.error(f"FATAL: Training dataset file not found at {CONFIG['DATA_PATH']}. Cannot proceed.")
            return
        except Exception as e:
            logging.exception(f"FATAL: Error during data loading or initial fitting of preprocessor/engineer: {e}")
            return
        model_loaded = detector.load_model()
        if not model_loaded or PERFORM_RETRAINING:
            if PERFORM_RETRAINING:
                logging.warning("Forcing model retraining as requested.")
            else:
                logging.info("No pre-trained model found or retraining requested. Starting training...")
            detector.train(X_train, y_train)
            if not detector.model:
                logging.error("FATAL: Model training finished, but detector.model is still None.")
                return
        else:
            logging.info("Pre-trained model loaded successfully.")
            if detector.feature_engineer.selected_features != feature_engineer.selected_features:
                logging.warning("Mismatch between features loaded from model checkpoint and features selected from current data fitting. Using features from the loaded model checkpoint.")
        if not detector.model:
            logging.error("FATAL: Detector model is not available after load/train attempt.")
            return
        if not preprocessor.feature_names:
            logging.error("FATAL: Preprocessor feature names were not set during fitting.")
            return
        if not feature_engineer.selected_features:
            logging.error("FATAL: Feature engineer features were not set during fitting/loading.")
            return
        if not detector.feature_engineer.selected_features:
            logging.error("FATAL: Detector's internal feature engineer features are not set.")
            return
        logging.info("All components initialized. Starting GUI...")
        gui = NIDSMonitorGUI(detector, alert_system, flow_monitor)
        gui.run()
        logging.info("NIDS Application finished.")
    except Exception as e:
        logging.exception(f"An unhandled exception occurred in main: {e}")

if __name__ == "__main__":
    is_admin = False
    try:
        if os.name == 'nt':
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        elif os.name == 'posix':
            is_admin = os.geteuid() == 0
    except AttributeError:
        logging.warning("Could not reliably determine administrative privileges.")
        is_admin = False
    except Exception as e:
        logging.error(f"Error checking admin privileges: {e}")
        is_admin = False
    if not is_admin:
        logging.warning("Script may not be running with root/administrator privileges. Packet sniffing might fail.")
        if os.name == 'nt':
            print("\n--- WARNING ---")
            print("This script likely needs administrator privileges to capture network packets.")
            print("Please try running it again by right-clicking and selecting 'Run as administrator'.")
            print("---------------\n")
    main()