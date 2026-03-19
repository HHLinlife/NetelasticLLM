"""
data/datasets/cic_darknet2020.py
CIC-Darknet2020 dataset: Tor / VPN / Non-VPN / Non-Tor traffic.

Reference:
    Lashkari et al. "DIDarknet: A Contemporary Approach to Detect and
    Characterize the Darknet Traffic using Deep Image Learning."
    ICCNS 2020.

Expected directory layout:
    root_dir/
        Darknet.csv            # or multiple CSV files
        NonTor.csv
        NonVPN.csv
        Tor.csv
        VPN.csv
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple

from .base_dataset import BaseTrafficDataset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Label catalogue                                                              #
# --------------------------------------------------------------------------- #
# Fine-grained: traffic type × application category
# Non-Tor subclasses
NON_TOR_CLASSES = [
    "NonTor_Audio-Streaming",
    "NonTor_Chat",
    "NonTor_Browsing",
    "NonTor_Email",
    "NonTor_File-Transfer",
    "NonTor_P2P",
    "NonTor_Video-Streaming",
]
# NonVPN subclasses
NON_VPN_CLASSES = [
    "NonVPN_Chat",
    "NonVPN_Audio-Streaming",
    "NonVPN_Email",
    "NonVPN_File-Transfer",
    "NonVPN_Video-Streaming",
    "NonVPN_VOIP",
]
# Tor subclasses (anonymity network — treated as malicious)
TOR_CLASSES = [
    "Tor_Audio-Streaming",
    "Tor_Browsing",
    "Tor_Chat",
    "Tor_File-Transfer",
    "Tor_Email",
    "Tor_P2P",
    "Tor_Video-Streaming",
    "Tor_VOIP",
]
# VPN subclasses (anonymity — treated as malicious)
VPN_CLASSES = [
    "VPN_File-Transfer",
    "VPN_Chat",
    "VPN_Audio-Streaming",
    "VPN_Email",
    "VPN_Video-Streaming",
    "VPN_VOIP",
]

ALL_CLASSES = NON_TOR_CLASSES + NON_VPN_CLASSES + TOR_CLASSES + VPN_CLASSES
LABEL2IDX   = {c: i for i, c in enumerate(ALL_CLASSES)}

# Coarse labels: benign = {NonTor, NonVPN}, malicious = {Tor, VPN}
BENIGN_CLASSES_SET  = set(NON_TOR_CLASSES + NON_VPN_CLASSES)
MALICIOUS_CLASSES_SET = set(TOR_CLASSES + VPN_CLASSES)


class CICDarknet2020Dataset(BaseTrafficDataset):
    """
    CIC-Darknet2020 dataset: 27 fine-grained application-layer classes
    across Non-Tor, NonVPN, Tor, and VPN network types.

    Coarse label: 0 = benign (Non-Tor / NonVPN), 1 = malicious (Tor / VPN).
    """

    def get_label_names(self) -> List[str]:
        return ALL_CLASSES

    def get_coarse_label(self, fine_label: int) -> int:
        name = ALL_CLASSES[fine_label]
        return 0 if name in BENIGN_CLASSES_SET else 1

    # ------------------------------------------------------------------ #

    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        csv_files = glob.glob(os.path.join(self.root_dir, "*.csv"))
        if csv_files:
            return self._load_csvs(csv_files)
        logger.warning(
            "CIC-Darknet2020: no CSVs found at %s — using synthetic data.",
            self.root_dir,
        )
        return self._generate_synthetic()

    def _load_csvs(self, csv_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        all_features, all_labels = [], []

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath, low_memory=False)
            except Exception as exc:
                logger.warning("Could not read %s: %s", fpath, exc)
                continue

            # CIC CSVs typically contain a 'Label' or 'label' column
            label_col = self._find_label_column(df)
            if label_col is None:
                logger.warning("No label column found in %s — skipping.", fpath)
                continue

            raw_labels = df[label_col].astype(str)
            df = df.drop(columns=[label_col])
            df = df.select_dtypes(include=[np.number])
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            for raw_label, row in zip(raw_labels, df.values):
                mapped = self._map_raw_label(raw_label)
                if mapped is None:
                    continue
                all_features.append(row.astype(np.float32))
                all_labels.append(LABEL2IDX[mapped])

        if not all_features:
            logger.warning("No valid data parsed — falling back to synthetic.")
            return self._generate_synthetic()

        features = np.stack(all_features, axis=0)
        labels   = np.array(all_labels, dtype=np.int64)
        features = self._align_feature_dim(features)
        logger.info("Loaded %d flows from CIC-Darknet2020.", len(features))
        return features, labels

    @staticmethod
    def _find_label_column(df: pd.DataFrame) -> str | None:
        for candidate in ["Label", "label", "CLASS", "class", "Category"]:
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _map_raw_label(raw: str) -> str | None:
        """Map raw string label (e.g. 'Tor_Chat') to ALL_CLASSES entry."""
        raw = raw.strip()
        if raw in LABEL2IDX:
            return raw
        # Try prefix matching for common format variants
        for cls in ALL_CLASSES:
            if cls.lower() == raw.lower():
                return cls
            if raw.lower().replace(" ", "_") == cls.lower():
                return cls
        return None

    def _align_feature_dim(self, features: np.ndarray) -> np.ndarray:
        n, d = features.shape
        if d >= self.feature_dim:
            return features[:, :self.feature_dim]
        pad = np.zeros((n, self.feature_dim - d), dtype=np.float32)
        return np.concatenate([features, pad], axis=1)

    @staticmethod
    def _generate_synthetic(
        n_per_class: int = 200,
        feature_dim: int = 83,
        random_seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_seed)
        feats, labs = [], []
        for idx in range(len(ALL_CLASSES)):
            mean = rng.uniform(-1.5, 1.5, feature_dim) + idx * 0.05
            X    = rng.normal(mean, 1.0, (n_per_class, feature_dim)).astype(np.float32)
            feats.append(X)
            labs.append(np.full(n_per_class, idx, dtype=np.int64))
        return np.concatenate(feats), np.concatenate(labs)
