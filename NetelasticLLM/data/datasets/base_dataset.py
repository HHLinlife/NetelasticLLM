"""
data/datasets/base_dataset.py
Abstract base class for all traffic datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import pickle
import logging

logger = logging.getLogger(__name__)


class BaseTrafficDataset(Dataset, ABC):
    """
    Abstract base class for encrypted traffic datasets.

    Each sample is a network flow represented as:
        x : feature vector (numpy array of shape [feature_dim])
        y_fine : fine-grained label (int)
        y_coarse : coarse label — 0=benign, 1=malicious (int)

    Subclasses must implement:
        - load_raw_data()
        - get_label_names()
        - get_coarse_label(fine_label) -> int
    """

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            features : np.ndarray of shape (N, feature_dim)
            labels   : np.ndarray of shape (N,) — fine-grained integer labels
        """
        raise NotImplementedError

    @abstractmethod
    def get_label_names(self) -> List[str]:
        """Returns list of fine-grained class names ordered by label index."""
        raise NotImplementedError

    @abstractmethod
    def get_coarse_label(self, fine_label: int) -> int:
        """Map fine-grained label to coarse binary label (0=benign,1=malicious)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Common implementation
    # ------------------------------------------------------------------

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        pretrain_ratio: float = 0.75,
        finetune_ratio: float = 0.20,
        test_ratio: float = 0.05,
        feature_dim: int = 83,
        normalize: bool = True,
        cache: bool = True,
        random_seed: int = 42,
    ):
        """
        Args:
            root_dir      : Path to dataset root directory.
            split         : One of 'pretrain', 'finetune', 'test'.
            pretrain_ratio: Fraction of data for pretraining.
            finetune_ratio: Fraction of data for fine-tuning.
            test_ratio    : Fraction of data for testing.
            feature_dim   : Expected feature vector dimensionality.
            normalize     : Whether to z-score normalise features.
            cache         : Cache processed data to disk.
            random_seed   : RNG seed for reproducible splitting.
        """
        assert split in ("pretrain", "finetune", "test"), \
            f"split must be one of pretrain/finetune/test, got {split}"
        assert abs(pretrain_ratio + finetune_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1."

        self.root_dir = root_dir
        self.split = split
        self.pretrain_ratio = pretrain_ratio
        self.finetune_ratio = finetune_ratio
        self.test_ratio = test_ratio
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.cache = cache
        self.random_seed = random_seed

        self.label_names = self.get_label_names()
        self.num_classes = len(self.label_names)

        # Load / restore from cache
        cache_path = os.path.join(root_dir, f"_cache_{split}.pkl")
        if cache and os.path.exists(cache_path):
            logger.info(f"Loading cached {split} split from {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.features = data["features"]
            self.fine_labels = data["fine_labels"]
            self.coarse_labels = data["coarse_labels"]
            self.mean = data.get("mean")
            self.std = data.get("std")
        else:
            self._build_dataset()
            if cache:
                os.makedirs(root_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "features": self.features,
                        "fine_labels": self.fine_labels,
                        "coarse_labels": self.coarse_labels,
                        "mean": self.mean,
                        "std": self.std,
                    }, f)
                logger.info(f"Cached {split} split to {cache_path}")

        logger.info(
            f"{self.__class__.__name__} [{split}] — "
            f"{len(self)} samples, {self.num_classes} classes"
        )

    def _build_dataset(self):
        """Load raw data, split, and optionally normalise."""
        features, fine_labels = self.load_raw_data()
        coarse_labels = np.array(
            [self.get_coarse_label(y) for y in fine_labels], dtype=np.int64
        )

        # Reproducible stratified split
        rng = np.random.default_rng(self.random_seed)
        n = len(features)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_pretrain = int(n * self.pretrain_ratio)
        n_finetune = int(n * self.finetune_ratio)

        split_idx = {
            "pretrain": idx[:n_pretrain],
            "finetune": idx[n_pretrain: n_pretrain + n_finetune],
            "test": idx[n_pretrain + n_finetune:],
        }[self.split]

        self.features = features[split_idx].astype(np.float32)
        self.fine_labels = fine_labels[split_idx].astype(np.int64)
        self.coarse_labels = coarse_labels[split_idx]

        # Normalisation (fit on pretrain only; apply to all)
        self.mean = None
        self.std = None
        if self.normalize:
            if self.split == "pretrain":
                self.mean = self.features.mean(axis=0)
                self.std = self.features.std(axis=0) + 1e-8
            else:
                # Caller must set mean/std after construction if needed
                pass
            if self.mean is not None:
                self.features = (self.features - self.mean) / self.std

    def apply_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Apply external normalization statistics (for finetune/test splits)."""
        self.mean = mean
        self.std = std
        self.features = (self.features - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "fine_label": torch.tensor(self.fine_labels[idx], dtype=torch.long),
            "coarse_label": torch.tensor(self.coarse_labels[idx], dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_class_distribution(self) -> Dict[str, int]:
        """Return per-class sample counts."""
        dist = {}
        for i, name in enumerate(self.label_names):
            dist[name] = int((self.fine_labels == i).sum())
        return dist

    def get_coarse_distribution(self) -> Dict[str, int]:
        return {
            "benign": int((self.coarse_labels == 0).sum()),
            "malicious": int((self.coarse_labels == 1).sum()),
        }

    def subset(self, n: int, random_seed: int = 0) -> "BaseTrafficDataset":
        """Return a shallow copy with at most n samples (for ablation)."""
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(self), min(n, len(self)), replace=False)
        obj = object.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.features = self.features[idx]
        obj.fine_labels = self.fine_labels[idx]
        obj.coarse_labels = self.coarse_labels[idx]
        return obj
