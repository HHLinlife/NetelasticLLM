"""
Flow Parser for Network Traffic Analysis

This module provides utilities for parsing PCAP files and extracting
flow-level representations suitable for LLM-based traffic analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Packet:
    """Represents a single network packet."""
    timestamp: float
    direction: int  # +1 for client->server, -1 for server->client
    length: int     # packet size in bytes
    payload_length: int  # payload size in bytes
    tcp_flags: Optional[Dict[str, bool]] = None
    

@dataclass
class Flow:
    """Represents a bidirectional network flow."""
    five_tuple: Tuple[str, int, str, int, str]  # (src_ip, src_port, dst_ip, dst_port, protocol)
    packets: List[Packet]
    start_time: float
    end_time: float
    
    def get_direction_sequence(self) -> List[int]:
        """Extract packet direction sequence."""
        return [p.direction for p in self.packets]
    
    def get_length_sequence(self) -> List[int]:
        """Extract packet length sequence."""
        return [p.length for p in self.packets]
    
    def get_timing_sequence(self) -> List[float]:
        """
        Extract inter-arrival time sequence.
        
        Returns:
            List of inter-arrival times in milliseconds.
            First element is 0 by convention (δ₁ = 0).
        """
        if not self.packets:
            return []
        
        timings = [0.0]  # δ₁ = 0
        for i in range(1, len(self.packets)):
            delta = (self.packets[i].timestamp - self.packets[i-1].timestamp) * 1000  # convert to ms
            timings.append(max(0.0, delta))  # ensure non-negative
        
        return timings
    
    def to_dict(self) -> Dict:
        """Convert flow to dictionary representation."""
        return {
            "direction": self.get_direction_sequence(),
            "length": self.get_length_sequence(),
            "timing": self.get_timing_sequence(),
            "duration": (self.end_time - self.start_time) * 1000,  # in ms
            "num_packets": len(self.packets),
            "total_bytes": sum(p.length for p in self.packets)
        }


class FlowParser:
    """
    Parser for converting PCAP files to flow representations.
    
    This parser implements:
    - 5-tuple flow identification
    - Bidirectional flow merging
    - Idle timeout-based flow termination
    - Direction assignment based on first packet
    """
    
    def __init__(
        self,
        idle_timeout: float = 60.0,
        min_packets: int = 2,
        min_payload_bytes: int = 1
    ):
        """
        Initialize flow parser.
        
        Args:
            idle_timeout: Flow idle timeout in seconds
            min_packets: Minimum packets for valid flow
            min_payload_bytes: Minimum payload bytes for valid flow
        """
        self.idle_timeout = idle_timeout
        self.min_packets = min_packets
        self.min_payload_bytes = min_payload_bytes
        
        # Flow tracking
        self.active_flows: Dict[Tuple, Flow] = {}
        self.completed_flows: List[Flow] = []
    
    def parse_pcap(self, pcap_path: str) -> List[Flow]:
        """
        Parse a PCAP file and extract flows.
        
        Args:
            pcap_path: Path to PCAP file
            
        Returns:
            List of Flow objects
        """
        try:
            from scapy.all import rdpcap, IP, TCP, UDP
        except ImportError:
            logger.error("scapy not installed. Install with: pip install scapy")
            raise
        
        logger.info(f"Parsing PCAP file: {pcap_path}")
        packets = rdpcap(pcap_path)
        
        for pkt in packets:
            self._process_packet(pkt)
        
        # Finalize all active flows
        self._finalize_all_flows()
        
        # Filter flows
        valid_flows = self._filter_flows()
        
        logger.info(f"Extracted {len(valid_flows)} valid flows from {len(packets)} packets")
        return valid_flows
    
    def _process_packet(self, pkt) -> None:
        """
        Process a single packet and update flow state.
        
        Args:
            pkt: Scapy packet object
        """
        from scapy.all import IP, TCP, UDP
        
        if not pkt.haslayer(IP):
            return
        
        ip_layer = pkt[IP]
        
        # Determine transport protocol
        if pkt.haslayer(TCP):
            transport = pkt[TCP]
            protocol = "TCP"
        elif pkt.haslayer(UDP):
            transport = pkt[UDP]
            protocol = "UDP"
        else:
            return
        
        # Extract 5-tuple
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        src_port = transport.sport
        dst_port = transport.dport
        
        # Create forward and reverse tuples
        forward_tuple = (src_ip, src_port, dst_ip, dst_port, protocol)
        reverse_tuple = (dst_ip, dst_port, src_ip, src_port, protocol)
        
        # Determine if this belongs to an existing flow
        timestamp = float(pkt.time)
        
        if forward_tuple in self.active_flows:
            flow = self.active_flows[forward_tuple]
            direction = 1  # client->server
        elif reverse_tuple in self.active_flows:
            flow = self.active_flows[reverse_tuple]
            direction = -1  # server->client
        else:
            # New flow - first packet determines direction
            flow = Flow(
                five_tuple=forward_tuple,
                packets=[],
                start_time=timestamp,
                end_time=timestamp
            )
            self.active_flows[forward_tuple] = flow
            direction = 1
        
        # Check idle timeout
        if timestamp - flow.end_time > self.idle_timeout:
            self._finalize_flow(forward_tuple if direction == 1 else reverse_tuple)
            # Start new flow
            flow = Flow(
                five_tuple=forward_tuple if direction == 1 else reverse_tuple,
                packets=[],
                start_time=timestamp,
                end_time=timestamp
            )
            self.active_flows[forward_tuple if direction == 1 else reverse_tuple] = flow
        
        # Add packet to flow
        packet = Packet(
            timestamp=timestamp,
            direction=direction,
            length=len(pkt),
            payload_length=len(pkt.payload) if hasattr(pkt, 'payload') else 0,
            tcp_flags=self._extract_tcp_flags(pkt) if pkt.haslayer(TCP) else None
        )
        
        flow.packets.append(packet)
        flow.end_time = timestamp
    
    def _extract_tcp_flags(self, pkt) -> Dict[str, bool]:
        """Extract TCP flags from packet."""
        from scapy.all import TCP
        
        tcp = pkt[TCP]
        return {
            "FIN": bool(tcp.flags & 0x01),
            "SYN": bool(tcp.flags & 0x02),
            "RST": bool(tcp.flags & 0x04),
            "PSH": bool(tcp.flags & 0x08),
            "ACK": bool(tcp.flags & 0x10),
            "URG": bool(tcp.flags & 0x20)
        }
    
    def _finalize_flow(self, flow_tuple: Tuple) -> None:
        """Move flow from active to completed."""
        if flow_tuple in self.active_flows:
            flow = self.active_flows.pop(flow_tuple)
            self.completed_flows.append(flow)
    
    def _finalize_all_flows(self) -> None:
        """Finalize all active flows."""
        for flow_tuple in list(self.active_flows.keys()):
            self._finalize_flow(flow_tuple)
    
    def _filter_flows(self) -> List[Flow]:
        """
        Filter flows based on validity constraints.
        
        Returns:
            List of valid flows
        """
        valid_flows = []
        
        for flow in self.completed_flows:
            # Check minimum packets
            if len(flow.packets) < self.min_packets:
                continue
            
            # Check minimum payload bytes
            total_payload = sum(p.payload_length for p in flow.packets)
            if total_payload < self.min_payload_bytes:
                continue
            
            valid_flows.append(flow)
        
        return valid_flows
    
    @staticmethod
    def flows_to_dataset_format(flows: List[Flow]) -> List[Dict]:
        """
        Convert flows to dataset-compatible format.
        
        Args:
            flows: List of Flow objects
            
        Returns:
            List of flow dictionaries
        """
        return [flow.to_dict() for flow in flows]
