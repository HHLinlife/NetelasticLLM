"""
data/datasets/cesnet_tls22.py
CESNET-TLS22: backbone-scale TLS service identification dataset.

Reference:
    Luxemburk & Čejka, "Fine-grained TLS services classification with
    reject option." Computer Networks 220 (2023).

Expected layout:
    root_dir/
        cesnet_tls22.csv      # single large CSV
        # OR per-service CSV files
        streaming/
        social/
        videoconferencing/
        filesharing/
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
#  Label catalogue (from paper, Table 6)                                       #
# --------------------------------------------------------------------------- #

STREAMING_CLASSES = [
    "twitch", "unpkg", "youtube", "facebook-media", "o2tv",
    "seznam-media", "amazon-prime", "netflix", "google-fonts",
    "font-awesome", "docker-registry", "super-media", "obalkyknih",
    "npm-registry", "vimeo", "alza-cdn",
]

SOCIAL_CLASSES = [
    "tiktok", "instagram", "snapchat", "tinder",
    "facebook-web", "twitter",
]

VIDEOCONF_CLASSES = [
    "skype", "teams", "zoom", "google-hangouts", "webex",
]

FILESHARE_CLASSES = [
    "office-365", "dropbox", "microsoft-onedrive", "github",
    "google-drive", "owncloud", "apple-icloud", "uschovna",
    "ulozto", "adobe-cloud",
]

ALL_CLASSES  = (STREAMING_CLASSES + SOCIAL_CLASSES +
                VIDEOCONF_CLASSES + FILESHARE_CLASSES)   # 37 classes
LABEL2IDX    = {c: i for i, c in enumerate(ALL_CLASSES)}

# All CESNET-TLS22 classes are benign (service identification task).
# For coarse labelling we follow paper convention: all = benign (0).
# The "malicious" concept is inherited from unknown/reject classes not in
# this corpus.  Here every known service maps to coarse=0.


class CESNETDataset(BaseTrafficDataset):
    """
    CESNET-TLS22 TLS service identification dataset.
    37 fine-grained service classes.  All coarse labels = 0 (benign).
    """

    def get_label_names(self) -> List[str]:
        return ALL_CLASSES

    def get_coarse_label(self, fine_label: int) -> int:
        # All CESNET classes are benign services
        return 0

    # ------------------------------------------------------------------ #

    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        single_csv = os.path.join(self.root_dir, "cesnet_tls22.csv")
        if os.path.exists(single_csv):
            return self._load_single_csv(single_csv)

        subdir_csvs = glob.glob(os.path.join(self.root_dir, "**", "*.csv"),
                                recursive=True)
        if subdir_csvs:
            return self._load_multi_csv(subdir_csvs)

        logger.warning(
            "CESNET-TLS22: no data found at %s — using synthetic data.",
            self.root_dir,
        )
        return self._generate_synthetic()

    def _load_single_csv(self, fpath: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(fpath, low_memory=False)
        label_col = self._detect_label_col(df)
        if label_col is None:
            raise ValueError(f"Cannot find label column in {fpath}")

        raw_labels = df[label_col].astype(str)
        df = df.drop(columns=[label_col])
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        features, labels = [], []
        for raw, row in zip(raw_labels, df.values):
            mapped = raw.strip().lower()
            if mapped not in LABEL2IDX:
                continue
            features.append(row.astype(np.float32))
            labels.append(LABEL2IDX[mapped])

        features = self._align_feature_dim(np.stack(features))
        labels   = np.array(labels, dtype=np.int64)
        logger.info("Loaded %d flows from CESNET-TLS22.", len(features))
        return features, labels

    def _load_multi_csv(self, paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        feats, labs = [], []
        for fpath in paths:
            service_name = os.path.splitext(os.path.basename(fpath))[0].lower()
            if service_name not in LABEL2IDX:
                continue
            try:
                df = pd.read_csv(fpath, low_memory=False)
            except Exception as exc:
                logger.warning("Cannot read %s: %s", fpath, exc)
                continue
            df = df.select_dtypes(include=[np.number])
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if df.empty:
                continue
            feats.append(df.values.astype(np.float32))
            labs.append(np.full(len(df), LABEL2IDX[service_name], dtype=np.int64))

        if not feats:
            return self._generate_synthetic()
        features = self._align_feature_dim(np.concatenate(feats))
        labels   = np.concatenate(labs)
        logger.info("Loaded %d flows from CESNET-TLS22.", len(features))
        return features, labels

    @staticmethod
    def _detect_label_col(df: pd.DataFrame):
        for c in ["Label", "label", "Service", "service", "class", "Class"]:
            if c in df.columns:
                return c
        return None

    def _align_feature_dim(self, features: np.ndarray) -> np.ndarray:
        n, d = features.shape
        if d >= self.feature_dim:
            return features[:, :self.feature_dim]
        return np.concatenate(
            [features, np.zeros((n, self.feature_dim - d), dtype=np.float32)],
            axis=1,
        )

    @staticmethod
    def _generate_synthetic(
        n_per_class: int = 150,
        feature_dim: int = 83,
        random_seed: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_seed)
        feats, labs = [], []
        for idx in range(len(ALL_CLASSES)):
            mu = rng.uniform(-1.0, 1.0, feature_dim)
            X  = rng.normal(mu, 0.8, (n_per_class, feature_dim)).astype(np.float32)
            feats.append(X)
            labs.append(np.full(n_per_class, idx, dtype=np.int64))
        return np.concatenate(feats), np.concatenate(labs)


# Dataset factory helper
def make_dataset(name: str, **kwargs) -> BaseTrafficDataset:
    """
    Convenience factory.
    name : 'ustc' | 'cic' | 'cesnet'
    """
    name = name.lower()
    if name in ("ustc", "ustc_tfc2016"):
        from .ustc_tfc2016 import USTCTFC2016Dataset
        return USTCTFC2016Dataset(**kwargs)
    elif name in ("cic", "cic_darknet2020"):
        from .cic_darknet2020 import CICDarknet2020Dataset
        return CICDarknet2020Dataset(**kwargs)
    elif name in ("cesnet", "cesnet_tls22"):
        return CESNETDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
