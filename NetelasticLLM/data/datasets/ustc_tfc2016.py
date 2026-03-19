"""
data/datasets/ustc_tfc2016.py
USTC-TFC2016 encrypted traffic dataset.

Reference:
    Huang et al. "BSTFNet: An Encrypted Malicious Traffic Classification
    Method Integrating Global Semantic and Spatiotemporal Features."
    Computers, Materials and Continua 78, 3 (2024).

Expected directory layout:
    root_dir/
        benign/
            BitTorrent.csv
            Facetime.csv
            ...
        malware/
            Cridex.csv
            Geodo.csv
            ...
Each CSV must contain a header row followed by one flow per row,
with the last column being the string class label.
All numeric feature columns come first.
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

BENIGN_CLASSES = [
    "BitTorrent", "Facetime", "FTP", "Gmail", "MySQL",
    "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft",
]

MALWARE_CLASSES = [
    "Cridex", "Geodo", "Htbot", "Miuref", "Neris",
    "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus",
]

ALL_CLASSES = BENIGN_CLASSES + MALWARE_CLASSES          # 20 fine-grained classes
LABEL2IDX    = {c: i for i, c in enumerate(ALL_CLASSES)}
BENIGN_IDX   = set(range(len(BENIGN_CLASSES)))           # indices 0-9
MALWARE_IDX  = set(range(len(BENIGN_CLASSES), len(ALL_CLASSES)))  # indices 10-19


# --------------------------------------------------------------------------- #
#  Dataset class                                                                #
# --------------------------------------------------------------------------- #

class USTCTFC2016Dataset(BaseTrafficDataset):
    """
    USTC-TFC2016 dataset with 10 benign and 10 malware classes.

    Usage
    -----
    >>> ds = USTCTFC2016Dataset(root_dir="data/ustc_tfc2016", split="finetune")
    >>> sample = ds[0]   # {'features': Tensor, 'fine_label': Tensor, 'coarse_label': Tensor}
    """

    def get_label_names(self) -> List[str]:
        return ALL_CLASSES

    def get_coarse_label(self, fine_label: int) -> int:
        return 0 if fine_label in BENIGN_IDX else 1

    # ------------------------------------------------------------------ #
    #  Data loading                                                         #
    # ------------------------------------------------------------------ #

    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse all CSV files under root_dir/benign/ and root_dir/malware/.
        If CSVs are absent, generate synthetic data for unit-testing.
        """
        benign_dir  = os.path.join(self.root_dir, "benign")
        malware_dir = os.path.join(self.root_dir, "malware")

        has_real = os.path.isdir(benign_dir) and os.path.isdir(malware_dir)

        if has_real:
            return self._load_csv_data(benign_dir, malware_dir)
        else:
            logger.warning(
                "USTC-TFC2016: real data not found at %s. "
                "Generating synthetic placeholder data.", self.root_dir
            )
            return self._generate_synthetic()

    def _load_csv_data(
        self,
        benign_dir: str,
        malware_dir: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read CSV files, infer numeric features, and assign integer labels."""
        all_features, all_labels = [], []

        for class_dir, class_list in [
            (benign_dir,  BENIGN_CLASSES),
            (malware_dir, MALWARE_CLASSES),
        ]:
            for class_name in class_list:
                pattern = os.path.join(class_dir, f"{class_name}*.csv")
                files   = glob.glob(pattern)
                if not files:
                    logger.debug("No CSV found for class %s at %s", class_name, pattern)
                    continue

                frames = []
                for fpath in files:
                    try:
                        df = pd.read_csv(fpath, low_memory=False)
                        frames.append(df)
                    except Exception as exc:
                        logger.warning("Could not read %s: %s", fpath, exc)

                if not frames:
                    continue

                df = pd.concat(frames, ignore_index=True)
                df = self._clean_dataframe(df)

                label_idx = LABEL2IDX[class_name]
                labels    = np.full(len(df), label_idx, dtype=np.int64)

                all_features.append(df.values.astype(np.float32))
                all_labels.append(labels)

        if not all_features:
            raise RuntimeError(
                f"No data loaded from {benign_dir} / {malware_dir}. "
                "Check directory structure."
            )

        features = np.concatenate(all_features, axis=0)
        labels   = np.concatenate(all_labels,   axis=0)

        # Truncate / pad feature dimension
        features = self._align_feature_dim(features)

        logger.info("Loaded %d flows from USTC-TFC2016.", len(features))
        return features, labels

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only numeric columns; replace NaN/Inf; drop label column."""
        # Drop non-numeric columns (e.g. IP, timestamp, label string)
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        return df

    def _align_feature_dim(self, features: np.ndarray) -> np.ndarray:
        """Ensure feature matrix has exactly self.feature_dim columns."""
        n, d = features.shape
        if d >= self.feature_dim:
            return features[:, :self.feature_dim]
        # Zero-pad if fewer features than expected
        pad = np.zeros((n, self.feature_dim - d), dtype=np.float32)
        return np.concatenate([features, pad], axis=1)

    @staticmethod
    def _generate_synthetic(
        n_per_class: int = 500,
        feature_dim: int = 83,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate class-conditional Gaussian data for smoke-testing.
        Each class has a unique mean offset so a classifier can learn.
        """
        rng = np.random.default_rng(random_seed)
        features_list, labels_list = [], []

        for idx, _ in enumerate(ALL_CLASSES):
            mean   = rng.uniform(-2.0, 2.0, feature_dim) + idx * 0.1
            cov    = np.eye(feature_dim) * rng.uniform(0.5, 1.5)
            X      = rng.multivariate_normal(mean, cov, n_per_class).astype(np.float32)
            y      = np.full(n_per_class, idx, dtype=np.int64)
            features_list.append(X)
            labels_list.append(y)

        return (
            np.concatenate(features_list, axis=0),
            np.concatenate(labels_list,   axis=0),
        )


# --------------------------------------------------------------------------- #
#  Standalone test                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = USTCTFC2016Dataset(root_dir="/tmp/ustc_test", split="pretrain", cache=False)
    print("Dataset length:", len(ds))
    print("Sample:", {k: v.shape for k, v in ds[0].items()})
    print("Class dist:", ds.get_class_distribution())
    print("Coarse dist:", ds.get_coarse_distribution())
