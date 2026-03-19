"""
data/preprocessing/flow_parser.py
Converts raw PCAP files into bidirectional flow records.

Each flow is a sequence of (direction, length, inter-arrival) triplets,
per Definition 1 in the paper:
    x = {(d_i, l_i, δ_i)}_{i=1}^{N}
where
    d_i ∈ {+1, -1}  — packet direction (+1 = client→server)
    l_i              — packet length in bytes
    δ_i = t_i - t_{i-1} — inter-arrival interval (δ_1 = 0)

Requires: scapy  (pip install scapy)
Falls back to a mock implementation when scapy is unavailable.
"""

from __future__ import annotations

import os
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import scapy; fall back gracefully
try:
    from scapy.all import PcapReader, IP, TCP, UDP         # type: ignore
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("scapy not found — PCAP parsing disabled. "
                   "Install with: pip install scapy")


# --------------------------------------------------------------------------- #
#  Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Packet:
    """A single observed packet."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int          # 6=TCP, 17=UDP
    length: int            # total packet length (bytes)


@dataclass
class Flow:
    """
    Bidirectional network flow derived from a 5-tuple key.

    Attributes
    ----------
    key          : (src_ip, dst_ip, src_port, dst_port, protocol)
    client_ip    : IP of the flow initiator (first seen src)
    directions   : List[int] — +1 or -1 per packet
    lengths      : List[int] — packet lengths
    timestamps   : List[float]
    deltas       : List[float] — inter-arrival intervals (δ_1 = 0)
    """
    key: Tuple
    client_ip: str = ""
    directions: List[int]  = field(default_factory=list)
    lengths: List[int]     = field(default_factory=list)
    timestamps: List[float]= field(default_factory=list)
    deltas: List[float]    = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Return (N, 3) array: columns = [direction, length, delta]."""
        return np.stack([
            np.array(self.directions, dtype=np.float32),
            np.array(self.lengths,    dtype=np.float32),
            np.array(self.deltas,     dtype=np.float32),
        ], axis=1)

    @property
    def num_packets(self) -> int:
        return len(self.directions)

    def is_valid(
        self,
        min_packets: int = 2,
        min_payload: int = 1,
    ) -> bool:
        return (
            self.num_packets >= min_packets and
            sum(self.lengths) >= min_payload
        )


# --------------------------------------------------------------------------- #
#  PCAP → Flow conversion                                                       #
# --------------------------------------------------------------------------- #

class FlowParser:
    """
    Parses a PCAP file and segments packets into bidirectional flows
    using a 5-tuple key with an idle-timeout policy.

    Parameters
    ----------
    idle_timeout   : Seconds of inactivity before a new flow is started.
    min_packets    : Minimum packets for a valid flow.
    max_packets    : Truncate flows longer than this.
    """

    def __init__(
        self,
        idle_timeout: float = 60.0,
        min_packets: int = 2,
        max_packets: int = 200,
    ):
        self.idle_timeout = idle_timeout
        self.min_packets  = min_packets
        self.max_packets  = max_packets

    # ------------------------------------------------------------------ #

    def parse_pcap(self, pcap_path: str) -> List[Flow]:
        """
        Parse a single PCAP file and return a list of valid flows.

        Falls back to an empty list with a warning when scapy is absent.
        """
        if not SCAPY_AVAILABLE:
            logger.error("scapy required for PCAP parsing.")
            return []

        if not os.path.isfile(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        active: Dict[str, Flow] = {}      # flow_key → current Flow
        last_seen: Dict[str, float] = {}  # flow_key → last packet time
        completed: List[Flow] = []

        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                parsed = self._extract_packet(pkt)
                if parsed is None:
                    continue

                # Canonical 5-tuple (sorted for bidirectional grouping)
                fwd_key = (
                    parsed.src_ip, parsed.dst_ip,
                    parsed.src_port, parsed.dst_port,
                    parsed.protocol,
                )
                rev_key = (
                    parsed.dst_ip, parsed.src_ip,
                    parsed.dst_port, parsed.src_port,
                    parsed.protocol,
                )

                # Prefer existing direction; otherwise create new
                if fwd_key in active:
                    flow_key = fwd_key
                    direction = +1
                elif rev_key in active:
                    flow_key = rev_key
                    direction = -1
                else:
                    # New flow
                    flow_key = fwd_key
                    direction = +1

                # Idle timeout check
                if flow_key in last_seen:
                    gap = parsed.timestamp - last_seen[flow_key]
                    if gap > self.idle_timeout:
                        completed.append(active.pop(flow_key))
                        last_seen.pop(flow_key)

                if flow_key not in active:
                    flow = Flow(key=flow_key, client_ip=parsed.src_ip)
                    active[flow_key] = flow

                flow = active[flow_key]
                if flow.num_packets < self.max_packets:
                    delta = (
                        parsed.timestamp - flow.timestamps[-1]
                        if flow.timestamps else 0.0
                    )
                    flow.directions.append(direction)
                    flow.lengths.append(parsed.length)
                    flow.timestamps.append(parsed.timestamp)
                    flow.deltas.append(delta)
                    last_seen[flow_key] = parsed.timestamp

        # Flush remaining active flows
        completed.extend(active.values())

        valid = [f for f in completed if f.is_valid(self.min_packets)]
        logger.info(
            "Parsed %s: %d total flows, %d valid.",
            pcap_path, len(completed), len(valid),
        )
        return valid

    def parse_directory(self, directory: str, label: int) -> List[Tuple[Flow, int]]:
        """Parse all PCAP files in a directory, returning (flow, label) pairs."""
        result = []
        for fname in sorted(os.listdir(directory)):
            if fname.endswith((".pcap", ".pcapng")):
                fpath = os.path.join(directory, fname)
                flows = self.parse_pcap(fpath)
                result.extend((f, label) for f in flows)
        return result

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_packet(pkt) -> Optional[Packet]:
        """Extract relevant fields from a scapy packet object."""
        if not (pkt.haslayer(IP) and (pkt.haslayer(TCP) or pkt.haslayer(UDP))):
            return None
        try:
            ip    = pkt[IP]
            layer = pkt[TCP] if pkt.haslayer(TCP) else pkt[UDP]
            return Packet(
                timestamp = float(pkt.time),
                src_ip    = ip.src,
                dst_ip    = ip.dst,
                src_port  = int(layer.sport),
                dst_port  = int(layer.dport),
                protocol  = 6 if pkt.haslayer(TCP) else 17,
                length    = len(pkt),
            )
        except Exception:
            return None


# --------------------------------------------------------------------------- #
#  Mock flow generator (for testing without PCAP files)                        #
# --------------------------------------------------------------------------- #

def generate_mock_flows(
    n_flows: int = 100,
    min_pkts: int = 5,
    max_pkts: int = 50,
    seed: int = 0,
) -> List[Flow]:
    """
    Generate synthetic Flow objects for unit tests and smoke runs.
    """
    rng = np.random.default_rng(seed)
    flows = []
    for i in range(n_flows):
        n = int(rng.integers(min_pkts, max_pkts + 1))
        directions  = rng.choice([1, -1], size=n).tolist()
        lengths     = rng.integers(40, 1461, size=n).tolist()
        timestamps  = np.cumsum(rng.exponential(0.1, size=n)).tolist()
        deltas      = [0.0] + [timestamps[j] - timestamps[j-1]
                               for j in range(1, n)]
        flow = Flow(
            key         = ("10.0.0.1", "10.0.0.2", 1234, 80, 6),
            client_ip   = "10.0.0.1",
            directions  = directions,
            lengths     = lengths,
            timestamps  = timestamps,
            deltas      = deltas,
        )
        flows.append(flow)
    return flows
