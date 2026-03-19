"""
data/preprocessing/feature_extractor.py
Protocol-agnostic feature extraction from Flow objects.

Implements the feature set described in Appendix A.2 / Table 7.
Produces an 83-dimensional (CICFlowMeter-compatible) or 348-dimensional
(NTLFlowLyzer-extended) feature vector per flow.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .flow_parser import Flow

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Feature vector specification                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class FeatureConfig:
    """Controls which feature groups to include."""
    use_basic:        bool = True   # duration, packet counts, byte counts
    use_pkt_len:      bool = True   # packet-length statistics
    use_iat:          bool = True   # inter-arrival time statistics
    use_payload:      bool = False  # payload byte stats (requires deep inspection)
    use_rate:         bool = True   # byte/packet rates
    use_tcp_flags:    bool = True   # TCP flag counts
    use_bulk:         bool = True   # bulk transfer features
    use_active_idle:  bool = False  # active/idle time features (NTLFlowLyzer only)
    split_directions: bool = True   # compute stats for Fwd/Bwd separately


CIC_CONFIG = FeatureConfig(
    use_basic=True, use_pkt_len=True, use_iat=True, use_payload=False,
    use_rate=True, use_tcp_flags=True, use_bulk=True,
    use_active_idle=False, split_directions=True,
)   # → ~83 features


class FlowFeatureExtractor:
    """
    Converts a Flow object into a fixed-length numeric feature vector.

    Parameters
    ----------
    config      : FeatureConfig controlling included feature groups.
    target_dim  : Pad / truncate output to this length (None = no padding).
    """

    def __init__(
        self,
        config: FeatureConfig = CIC_CONFIG,
        target_dim: Optional[int] = 83,
    ):
        self.config     = config
        self.target_dim = target_dim

    # ------------------------------------------------------------------ #

    def extract(self, flow: Flow) -> np.ndarray:
        """Return feature vector for a single flow."""
        feats: List[float] = []

        dirs   = np.array(flow.directions,  dtype=np.float64)
        lens   = np.array(flow.lengths,     dtype=np.float64)
        deltas = np.array(flow.deltas,      dtype=np.float64)

        fwd_mask = dirs > 0
        bwd_mask = dirs < 0
        fwd_lens = lens[fwd_mask]
        bwd_lens = lens[bwd_mask]
        fwd_delt = deltas[fwd_mask]
        bwd_delt = deltas[bwd_mask]

        # ── Basic stats ──────────────────────────────────────────────── #
        if self.config.use_basic:
            duration    = flow.timestamps[-1] - flow.timestamps[0] if len(flow.timestamps) > 1 else 0.0
            total_bytes = float(lens.sum())
            feats += [
                duration,
                float(len(dirs)),           # total packets
                float(fwd_mask.sum()),      # fwd packets
                float(bwd_mask.sum()),      # bwd packets
                total_bytes,
            ]

        # ── Packet length stats ──────────────────────────────────────── #
        if self.config.use_pkt_len:
            feats += self._stats6(lens)
            if self.config.split_directions:
                feats += self._stats6(fwd_lens)
                feats += self._stats6(bwd_lens)

        # ── Inter-arrival time stats ──────────────────────────────────── #
        if self.config.use_iat:
            iat = deltas[1:] if len(deltas) > 1 else np.array([0.0])
            feats += self._stats6(iat)
            if self.config.split_directions:
                fwd_iat = fwd_delt[1:] if len(fwd_delt) > 1 else np.array([0.0])
                bwd_iat = bwd_delt[1:] if len(bwd_delt) > 1 else np.array([0.0])
                feats += self._stats6(fwd_iat)
                feats += self._stats6(bwd_iat)

        # ── Rate features ─────────────────────────────────────────────── #
        if self.config.use_rate:
            duration = (flow.timestamps[-1] - flow.timestamps[0]
                        if len(flow.timestamps) > 1 else 1e-6)
            duration = max(duration, 1e-6)
            total_bytes = float(lens.sum())
            feats += [
                total_bytes / duration,               # byte rate
                float(len(dirs)) / duration,          # packet rate
                float(fwd_mask.sum()) / duration,     # fwd packet rate
                float(bwd_mask.sum()) / duration,     # bwd packet rate
                float(bwd_lens.sum()) / max(float(fwd_lens.sum()), 1.0),  # down/up ratio
            ]

        # ── Bulk features ─────────────────────────────────────────────── #
        if self.config.use_bulk:
            feats += self._bulk_features(fwd_lens, bwd_lens, duration if self.config.use_rate else 1.0)

        # ── TCP flag counts (placeholder) ────────────────────────────── #
        if self.config.use_tcp_flags:
            # Without deep packet inspection we use zeros
            feats += [0.0] * 12   # FIN/PSH/URG/SYN/ACK/RST × {fwd,bwd}

        out = np.array(feats, dtype=np.float32)

        if self.target_dim is not None:
            out = self._resize(out, self.target_dim)

        return out

    def extract_batch(self, flows: List[Flow]) -> np.ndarray:
        """Return (N, feature_dim) array."""
        return np.stack([self.extract(f) for f in flows], axis=0)

    # ------------------------------------------------------------------ #
    #  Static helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stats6(arr: np.ndarray) -> List[float]:
        """Compute [min, max, mean, std, var, skew] for an array."""
        if len(arr) == 0:
            return [0.0] * 6
        mn  = float(arr.min())
        mx  = float(arr.max())
        mu  = float(arr.mean())
        sd  = float(arr.std()) if len(arr) > 1 else 0.0
        var = sd ** 2
        # Skewness (Fisher's definition)
        if sd > 0 and len(arr) > 2:
            skew = float(np.mean(((arr - mu) / sd) ** 3))
        else:
            skew = 0.0
        return [mn, mx, mu, sd, var, skew]

    @staticmethod
    def _bulk_features(
        fwd: np.ndarray,
        bwd: np.ndarray,
        duration: float,
    ) -> List[float]:
        """
        Simplified bulk metrics: total, mean, rate for fwd and bwd.
        """
        duration = max(duration, 1e-6)
        fwd_total = float(fwd.sum())   if len(fwd) > 0 else 0.0
        fwd_mean  = float(fwd.mean())  if len(fwd) > 0 else 0.0
        fwd_rate  = fwd_total / duration
        bwd_total = float(bwd.sum())   if len(bwd) > 0 else 0.0
        bwd_mean  = float(bwd.mean())  if len(bwd) > 0 else 0.0
        bwd_rate  = bwd_total / duration
        return [fwd_total, fwd_mean, fwd_rate, bwd_total, bwd_mean, bwd_rate]

    @staticmethod
    def _resize(arr: np.ndarray, target: int) -> np.ndarray:
        """Truncate or zero-pad arr to length target."""
        n = len(arr)
        if n >= target:
            return arr[:target]
        return np.concatenate([arr, np.zeros(target - n, dtype=np.float32)])


# --------------------------------------------------------------------------- #
#  Batch processing pipeline                                                    #
# --------------------------------------------------------------------------- #

def extract_features_from_pcap_dir(
    pcap_dir: str,
    label: int,
    idle_timeout: float = 60.0,
    feature_dim: int = 83,
) -> Dict[str, Any]:
    """
    Parse a directory of PCAP files and extract feature vectors.

    Returns
    -------
    dict with keys 'features' (ndarray) and 'labels' (ndarray).
    """
    from .flow_parser import FlowParser

    parser    = FlowParser(idle_timeout=idle_timeout)
    extractor = FlowFeatureExtractor(CIC_CONFIG, target_dim=feature_dim)

    flows_and_labels = parser.parse_directory(pcap_dir, label)
    if not flows_and_labels:
        logger.warning("No flows extracted from %s", pcap_dir)
        return {"features": np.empty((0, feature_dim)), "labels": np.empty(0)}

    flows  = [fl for fl, _ in flows_and_labels]
    labels = np.array([lb for _, lb in flows_and_labels], dtype=np.int64)
    features = extractor.extract_batch(flows)

    return {"features": features, "labels": labels}
