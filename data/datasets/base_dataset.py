"""
Base Dataset Class for Encrypted Traffic Analysis

This module provides the abstract base class for all traffic datasets,
defining the common interface and preprocessing pipelines.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseTrafficDataset(Dataset, ABC):
    """
    Abstract base class for encrypted traffic datasets.
    
    All traffic datasets should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 256,
        min_packets: int = 2,
        min_payload_bytes: int = 1,
        preprocess: bool = True
    ):
        """
        Initialize base dataset.
        
        Args:
            data_path: Path to the dataset
            split: Data split (train, val, or test)
            max_sequence_length: Maximum number of packets per flow
            min_packets: Minimum packets required for valid flow
            min_payload_bytes: Minimum payload bytes for valid flow
            preprocess: Whether to apply preprocessing
        """
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.min_packets = min_packets
        self.min_payload_bytes = min_payload_bytes
        
        # Data containers
        self.flows = []
        self.labels = []
        self.coarse_labels = []
        self.fine_labels = []
        
        # Label mappings
        self.coarse_label_map = {"benign": 0, "malicious": 1}
        self.fine_label_map = {}
        
        # Load and preprocess data
        self._load_data()
        if preprocess:
            self._preprocess()
            
        logger.info(f"Loaded {len(self)} samples for {split} split")
    
    @abstractmethod
    def _load_data(self):
        """
        Load raw data from disk.
        Must be implemented by subclasses.
        """
        pass
    
    def _preprocess(self):
        """
        Apply preprocessing to loaded data.
        """
        logger.info("Preprocessing flows...")
        
        valid_indices = []
        for idx, flow in enumerate(self.flows):
            if self._is_valid_flow(flow):
                valid_indices.append(idx)
        
        # Filter invalid flows
        self.flows = [self.flows[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.coarse_labels = [self.coarse_labels[i] for i in valid_indices]
        self.fine_labels = [self.fine_labels[i] for i in valid_indices]
        
        logger.info(f"Retained {len(valid_indices)}/{len(self.flows) + len(valid_indices)} valid flows")
    
    def _is_valid_flow(self, flow: Dict) -> bool:
        """
        Check if a flow satisfies validity constraints.
        
        Args:
            flow: Flow dictionary with keys (direction, length, timing)
            
        Returns:
            True if flow is valid, False otherwise
        """
        num_packets = len(flow["direction"])
        total_bytes = sum(flow["length"])
        
        return (
            num_packets >= self.min_packets and
            total_bytes >= self.min_payload_bytes
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.flows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - direction: Packet direction sequence [N]
                - length: Packet length sequence [N]
                - timing: Inter-arrival time sequence [N]
                - coarse_label: Binary label (benign/malicious)
                - fine_label: Fine-grained class label
                - mask: Padding mask [max_seq_len]
        """
        flow = self.flows[idx]
        
        # Extract sequences
        direction = np.array(flow["direction"], dtype=np.int64)
        length = np.array(flow["length"], dtype=np.float32)
        timing = np.array(flow["timing"], dtype=np.float32)
        
        # Truncate or pad to max_sequence_length
        num_packets = len(direction)
        if num_packets > self.max_sequence_length:
            direction = direction[:self.max_sequence_length]
            length = length[:self.max_sequence_length]
            timing = timing[:self.max_sequence_length]
            num_packets = self.max_sequence_length
        
        # Create padding mask (1 for valid tokens, 0 for padding)
        mask = np.zeros(self.max_sequence_length, dtype=np.float32)
        mask[:num_packets] = 1.0
        
        # Pad sequences
        padded_direction = np.zeros(self.max_sequence_length, dtype=np.int64)
        padded_length = np.zeros(self.max_sequence_length, dtype=np.float32)
        padded_timing = np.zeros(self.max_sequence_length, dtype=np.float32)
        
        padded_direction[:num_packets] = direction
        padded_length[:num_packets] = length
        padded_timing[:num_packets] = timing
        
        return {
            "direction": torch.from_numpy(padded_direction),
            "length": torch.from_numpy(padded_length),
            "timing": torch.from_numpy(padded_timing),
            "coarse_label": torch.tensor(self.coarse_labels[idx], dtype=torch.long),
            "fine_label": torch.tensor(self.fine_labels[idx], dtype=torch.long),
            "mask": torch.from_numpy(mask)
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        all_lengths = []
        all_timings = []
        num_packets_list = []
        
        for flow in self.flows:
            all_lengths.extend(flow["length"])
            all_timings.extend(flow["timing"])
            num_packets_list.append(len(flow["direction"]))
        
        return {
            "num_samples": len(self),
            "avg_packets_per_flow": np.mean(num_packets_list),
            "std_packets_per_flow": np.std(num_packets_list),
            "avg_packet_length": np.mean(all_lengths),
            "std_packet_length": np.std(all_lengths),
            "avg_inter_arrival": np.mean(all_timings),
            "std_inter_arrival": np.std(all_timings),
            "coarse_class_distribution": self._get_class_distribution(self.coarse_labels),
            "fine_class_distribution": self._get_class_distribution(self.fine_labels)
        }
    
    def _get_class_distribution(self, labels: List[int]) -> Dict[int, float]:
        """Compute class distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        return {int(cls): count / total for cls, count in zip(unique, counts)}
